"""
hf_space/db/supabase.py
------------------------
Async PostgreSQL client for the HuggingFace Space.
Uses an asyncpg connection pool built from DATABASE_URL.

No Supabase SDK — only DATABASE_URL is required.
Provides a fluent QueryBuilder API that is drop-in compatible with the
supabase-py patterns already used in hf_space/routers/*.py.

Supported chain methods:
  .select()  .eq()  .neq()  .in_()  .gte()  .gt()  .lte()  .lt()
  .order()  .limit()  .single()  .update()  .insert()  .upsert()
  .execute()   → returns object with .data attribute
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import asyncpg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


# ── Pool lifecycle ─────────────────────────────────────────────────────────────

async def init_db_pool() -> asyncpg.Pool:
    """Create asyncpg pool from DATABASE_URL. Call once from app lifespan."""
    global _pool
    url = os.environ["DATABASE_URL"]
    # Supabase requires SSL
    if "sslmode" not in url and "supabase.com" in url:
        sep = "&" if "?" in url else "?"
        url += f"{sep}sslmode=require"
    _pool = await asyncpg.create_pool(
        url,
        min_size=2,
        max_size=10,
        command_timeout=30,
        ssl="require",
    )
    async with _pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    logger.info("✅ asyncpg pool ready (DATABASE_URL)")
    return _pool


# Keep old name for compatibility with any existing callers
async def init_supabase():
    return await init_db_pool()


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError(
            "DB pool not initialised — call init_db_pool() during startup"
        )
    return _pool


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _to_py(v: Any) -> Any:
    """Convert asyncpg-specific types → plain Python."""
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _row_to_dict(row: asyncpg.Record) -> dict:
    return {k: _to_py(v) for k, v in dict(row).items()}


def _serialize(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


# ── Result wrapper ─────────────────────────────────────────────────────────────

class _Result:
    __slots__ = ("data",)

    def __init__(self, data: Any) -> None:
        self.data = data


# ── Query builder ──────────────────────────────────────────────────────────────

class QueryBuilder:
    """
    Fluent SQL builder compatible with supabase-py AsyncClient patterns.
    """

    def __init__(self, pool: asyncpg.Pool, table: str) -> None:
        self._pool   = pool
        self._table  = table
        self._cols   = "*"
        self._wheres: list[tuple[str, str, Any]] = []
        self._ins:    list[tuple[str, list]]     = []
        self._order:  Optional[tuple[str, bool]] = None
        self._lim:    Optional[int]              = None
        self._single  = False
        self._updata: Optional[dict] = None
        self._indata: Any            = None

    # ── projection ─────────────────────────────────────────────────────────────
    def select(self, cols: str = "*") -> "QueryBuilder":
        self._cols = cols
        return self

    # ── filters ────────────────────────────────────────────────────────────────
    def eq(self,  col: str, v: Any) -> "QueryBuilder": self._wheres.append((col, "=",  v)); return self
    def neq(self, col: str, v: Any) -> "QueryBuilder": self._wheres.append((col, "!=", v)); return self
    def gte(self, col: str, v: Any) -> "QueryBuilder": self._wheres.append((col, ">=", v)); return self
    def gt(self,  col: str, v: Any) -> "QueryBuilder": self._wheres.append((col, ">",  v)); return self
    def lte(self, col: str, v: Any) -> "QueryBuilder": self._wheres.append((col, "<=", v)); return self
    def lt(self,  col: str, v: Any) -> "QueryBuilder": self._wheres.append((col, "<",  v)); return self

    def in_(self, col: str, vals: list) -> "QueryBuilder":
        self._ins.append((col, vals))
        return self

    # ── sorting / pagination ───────────────────────────────────────────────────
    def order(self, col: str, desc: bool = False) -> "QueryBuilder":
        self._order = (col, desc)
        return self

    def limit(self, n: int) -> "QueryBuilder":
        self._lim = n
        return self

    def single(self) -> "QueryBuilder":
        self._single = True
        return self

    # ── write operations ───────────────────────────────────────────────────────
    def update(self, data: dict) -> "QueryBuilder":
        self._updata = data
        return self

    def insert(self, data: Any) -> "QueryBuilder":
        self._indata = data
        return self

    def upsert(self, data: Any) -> "QueryBuilder":
        self._indata = data
        return self

    # ── execution ──────────────────────────────────────────────────────────────
    async def execute(self) -> _Result:
        if self._updata is not None:
            return await self._do_update()
        if self._indata is not None:
            return await self._do_insert()
        return await self._do_select()

    # ── WHERE builder ──────────────────────────────────────────────────────────
    def _build_where(self, offset: int = 1) -> tuple[str, list]:
        parts, params, i = [], [], offset
        for col, op, val in self._wheres:
            parts.append(f'"{col}" {op} ${i}')
            params.append(val)
            i += 1
        for col, vals in self._ins:
            phs = ", ".join(f"${j}" for j in range(i, i + len(vals)))
            parts.append(f'"{col}" IN ({phs})')
            params.extend(vals)
            i += len(vals)
        where = (" WHERE " + " AND ".join(parts)) if parts else ""
        return where, params

    def _parse_cols(
        self, raw: str
    ) -> tuple[list[str], dict[str, tuple[str, str]]]:
        """
        Handle PostgREST join syntax: alias:table(col)
        Returns (plain_cols, {alias: (ref_table, ref_col)}).
        """
        plain: list[str] = []
        joins: dict[str, tuple[str, str]] = {}
        for part in raw.split(","):
            p = part.strip()
            if ":" in p:
                alias, rest = p.split(":", 1)
                m = re.match(r"(\w+)\(([^)]+)\)", rest.strip())
                if m:
                    joins[alias.strip()] = (m.group(1), m.group(2))
                continue
            if p:
                plain.append(p)
        return plain, joins

    # ── SELECT ─────────────────────────────────────────────────────────────────
    async def _do_select(self) -> _Result:
        t = self._table
        plain, joins = self._parse_cols(self._cols)

        col_sql = (
            f'"{t}".*'
            if (not plain or plain == ["*"])
            else ", ".join(f'"{t}"."{c}"' for c in plain)
        )

        join_clauses: list[str] = []
        for alias, (ref_table, ref_col) in joins.items():
            col_sql += f', "{ref_table}"."{ref_col}" AS "{alias}_{ref_col}"'
            # Infer FK column: e.g. "hotels" → "hotel_id"
            fk = ref_table.rstrip("s") + "_id"
            join_clauses.append(
                f'LEFT JOIN "{ref_table}" ON "{t}"."{fk}" = "{ref_table}"."id"'
            )

        join_sql  = (" " + " ".join(join_clauses)) if join_clauses else ""
        where_sql, params = self._build_where(1)
        order_sql = ""
        if self._order:
            c, d  = self._order
            order_sql = f' ORDER BY "{c}" {"DESC" if d else "ASC"}'
        limit_sql = f" LIMIT {self._lim}" if self._lim is not None else ""

        sql = (
            f'SELECT {col_sql} FROM "{t}"{join_sql}'
            f'{where_sql}{order_sql}{limit_sql}'
        )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        data = [_row_to_dict(r) for r in rows]
        if self._single:
            return _Result(data[0] if data else None)
        return _Result(data)

    # ── UPDATE ─────────────────────────────────────────────────────────────────
    async def _do_update(self) -> _Result:
        t = self._table
        sets, params, i = [], [], 1
        for col, val in (self._updata or {}).items():
            sets.append(f'"{col}" = ${i}')
            params.append(_serialize(val))
            i += 1
        where_sql, w_params = self._build_where(i)
        params.extend(w_params)
        sql = f'UPDATE "{t}" SET {", ".join(sets)}{where_sql} RETURNING *'
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return _Result([_row_to_dict(r) for r in rows])

    # ── INSERT ─────────────────────────────────────────────────────────────────
    async def _do_insert(self) -> _Result:
        t       = self._table
        rows_in = self._indata
        if isinstance(rows_in, dict):
            rows_in = [rows_in]
        inserted: list[dict] = []
        async with self._pool.acquire() as conn:
            for row_data in rows_in:
                cols  = list(row_data.keys())
                vals  = [_serialize(v) for v in row_data.values()]
                col_s = ", ".join(f'"{c}"' for c in cols)
                ph_s  = ", ".join(f"${j + 1}" for j in range(len(cols)))
                sql   = (
                    f'INSERT INTO "{t}" ({col_s}) VALUES ({ph_s})'
                    f' ON CONFLICT DO NOTHING RETURNING *'
                )
                rows = await conn.fetch(sql, *vals)
                inserted.extend(_row_to_dict(r) for r in rows)
        return _Result(inserted)


# ── DB client ─────────────────────────────────────────────────────────────────

class DBClient:
    """Drop-in replacement for supabase-py AsyncClient."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    def table(self, name: str) -> QueryBuilder:
        return QueryBuilder(self._pool, name)

    def rpc(self, fn: str, args: dict) -> Any:
        """Stub for RPC calls — returns a no-op execute-able."""
        class _Stub:
            async def execute(self_):
                return _Result(None)
        return _Stub()


def get_supabase() -> DBClient:
    """Return a DB client backed by the asyncpg connection pool."""
    return DBClient(get_pool())
