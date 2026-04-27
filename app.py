from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_file, render_template, session, redirect, url_for
from pymongo import MongoClient, ReturnDocument
try:
    import mongomock  # type: ignore
except Exception:
    mongomock = None

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

HF_HOME_DIR = PROJECT_DIR / ".hf_home"
HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR.resolve()))

PYDEPS_DIR = PROJECT_DIR / ".pydeps"
if PYDEPS_DIR.exists():
    sys.path.insert(0, str(PYDEPS_DIR.resolve()))


MONGO_URL = os.environ.get("MONGO_URL", "").strip()
if MONGO_URL:
    client = MongoClient(MONGO_URL)
else:
    if mongomock is not None:
        client = mongomock.MongoClient()
    else:
        client = MongoClient("mongodb://localhost:27017")
db = client["tejas_kitchen"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def money(cents: int) -> str:
    return f"{cents / 100:.2f}"


def order_totals(order: Dict[str, Any]) -> Tuple[int, int, int]:
    subtotal = 0
    tax = 0
    for it in order.get("items", []):
        item_total = int(it.get("qty", 1)) * int(it.get("price_cents", 0))
        subtotal += item_total
        item_tax = item_total * (int(it.get("gst_percent", 0)) / 100.0)
        tax += int(round(item_tax))
    total = subtotal + tax
    return subtotal, tax, total


def ensure_seed_data() -> None:
    if db["menu_items"].count_documents({}) > 0:
        return
    seq = 1
    items = [
        {"id": seq, "name": "Masala Dosa", "category": "main", "price_cents": 8000, "gst_percent": 5, "available": True},
        {"id": seq + 1, "name": "Idli", "category": "main", "price_cents": 4000, "gst_percent": 5, "available": True},
        {"id": seq + 2, "name": "Veg Biryani", "category": "main", "price_cents": 12000, "gst_percent": 5, "available": True},
        {"id": seq + 3, "name": "Coffee", "category": "drinks", "price_cents": 3000, "gst_percent": 5, "available": True},
        {"id": seq + 4, "name": "Tea", "category": "drinks", "price_cents": 2500, "gst_percent": 5, "available": True},
        {"id": seq + 5, "name": "Water Bottle", "category": "drinks", "price_cents": 2000, "gst_percent": 5, "available": True},
    ]
    db["menu_items"].insert_many(items)


ensure_seed_data()

app = Flask(__name__)
app.secret_key = "tejas_kitchen_secret_key"


def parse_quantity(token: str) -> Optional[int]:
    t = token.strip().lower()
    if t.isdigit():
        n = int(t)
        if 1 <= n <= 20:
            return n
        return None
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return words.get(t)


def tokenize_text(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9']+", text.lower()) if t]


class MenuLike:
    def __init__(self, d: Dict[str, Any]):
        self.id = int(d.get("id", 0))
        self.name = str(d.get("name", ""))
        self.category = str(d.get("category", ""))
        self.price_cents = int(d.get("price_cents", 0))
        self.gst_percent = int(d.get("gst_percent", 5))


def nlu_extract_order(transcript: str, menu: List[MenuLike]) -> List[Dict[str, Any]]:
    toks = tokenize_text(transcript)
    if not toks:
        return []

    # Prepare menu items: tokenized and sorted by length (descending)
    # This ensures "Masala Dosa" is matched before "Dosa"
    menu_prepared = []
    for m in menu:
        m_toks = tokenize_text(m.name)
        menu_prepared.append((m, m_toks))
    
    menu_prepared.sort(key=lambda x: len(x[1]), reverse=True)

    results: List[Dict[str, Any]] = []
    pending_qty = 1
    
    i = 0
    while i < len(toks):
        # 1. Check for quantity
        qty = parse_quantity(toks[i])
        if qty is not None:
            pending_qty = qty
            i += 1
            continue
            
        # 2. Check for menu item match
        match = None
        match_len = 0
        
        for m, m_toks in menu_prepared:
            # Check if next tokens match this menu item
            if i + len(m_toks) <= len(toks):
                if toks[i : i + len(m_toks)] == m_toks:
                    match = m
                    match_len = len(m_toks)
                    break
        
        if match:
            results.append({
                "menu_item_id": match.id,
                "name": match.name,
                "qty": pending_qty,
                "notes": ""
            })
            pending_qty = 1 # Reset quantity after use
            i += match_len
            continue
            
        # 3. Filler word, skip
        i += 1

    return results


@app.get("/")
def index() -> Response:
    cid = session.get("customer_id")
    if not cid:
        return redirect(url_for("customer_form"))
    customer = db["customers"].find_one({"id": int(cid)}) or {}
    return render_template("index.html", customer=customer)


@app.route("/customer", methods=["GET", "POST"])
def customer_form() -> Response:
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        phone = (request.form.get("phone") or "").strip()
        if not name or not phone:
            return render_template("customer.html", error="Please enter name and phone.")
        existing = db["customers"].find_one({"phone": phone})
        if existing:
            session["customer_id"] = int(existing.get("id"))
            return redirect(url_for("index"))
        next_id = db["counters"].find_one_and_update(
            {"_id": "customers"},
            {"$inc": {"seq": 1}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )["seq"]
        db["customers"].insert_one({"id": next_id, "name": name, "phone": phone, "created_at": utcnow()})
        session["customer_id"] = int(next_id)
        return redirect(url_for("index"))
    return render_template("customer.html")


@app.get("/customer/<int:customer_id>")
def customer_details(customer_id: int) -> Response:
    cust = db["customers"].find_one({"id": customer_id})
    if not cust:
        return Response("not found", status=404)
    o = db["orders"].find_one({"customer.id": customer_id, "status": "paid"}, sort=[("created_at", -1)])
    if not o:
        # If no paid order, try finding any order
        o = db["orders"].find_one({"customer.id": customer_id}, sort=[("created_at", -1)])
    
    if not o:
        return render_template("customer_details.html", customer=cust, order=None, taxes_list=[], subtotal_fmt="0.00", total_fmt="0.00")
    
    subtotal, tax, total = order_totals(o)
    # Build items with unit price and line total
    items_list = []
    tax_breakdown: Dict[int, int] = {}
    for it in o.get("items", []):
        price = int(it.get("price_cents", 0))
        qty = int(it.get("qty", 1))
        gst = int(it.get("gst_percent", 0))
        line_sub = price * qty
        line_tax = int(round(line_sub * gst / 100.0))
        tax_breakdown[gst] = tax_breakdown.get(gst, 0) + line_tax
        items_list.append({
            "name": it.get("name"),
            "qty": qty,
            "price_fmt": money(price),
            "line_total_fmt": money(line_sub + line_tax),
        })
    taxes_list = [{"rate": r, "amount_fmt": money(tax_breakdown[r])} for r in sorted(tax_breakdown.keys())]
    order_ctx = {
        "id": o.get("id"),
        "daily_id": o.get("daily_id") or o.get("id"),
        "status": o.get("status"),
        "time": o.get("created_at").strftime("%Y-%m-%d %H:%M") if isinstance(o.get("created_at"), datetime) else "",
        "line_items": items_list,
    }
    return render_template(
        "customer_details.html",
        customer=cust,
        order=order_ctx,
        taxes_list=taxes_list,
        subtotal_fmt=money(subtotal),
        total_fmt=money(total),
    )


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login() -> Response:
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "teja" and password == "teja":
            session["admin_user"] = username
            return redirect(url_for("admin_dashboard"))
        else:
            return render_template("admin_login.html", error="Invalid credentials")
    return render_template("admin_login.html")


@app.route("/admin")
def admin_dashboard() -> Response:
    if "admin_user" not in session:
        return redirect(url_for("admin_login"))

    now = utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    orders_coll = db["orders"]
    users_coll = db["users"]
    today_orders = orders_coll.count_documents({"created_at": {"$gte": today_start}})
    month_orders = orders_coll.count_documents({"created_at": {"$gte": month_start}})
    today_paid = orders_coll.find({"created_at": {"$gte": today_start}, "payment.status": "paid"})
    month_paid = orders_coll.find({"created_at": {"$gte": month_start}, "payment.status": "paid"})
    today_revenue_cents = sum(int(o.get("payment", {}).get("amount_cents", 0)) for o in today_paid)
    month_revenue_cents = sum(int(o.get("payment", {}).get("amount_cents", 0)) for o in month_paid)
    users = list(users_coll.find({}, sort=[("username", 1)]))
    users_data = [{"id": u.get("id"), "username": u.get("username"), "role": u.get("role"), "shift": u.get("shift")} for u in users]
    recent_orders = list(orders_coll.find({}, sort=[("created_at", -1)], limit=50))
    orders_data = []
    for o in recent_orders:
        subtotal, tax, total = order_totals(o)
        orders_data.append({
            "id": o.get("id"),
            "daily_id": o.get("daily_id") or o.get("id"),
            "table_id": o.get("table_id"),
            "status": o.get("status"),
            "total": money(total),
            "order_items": [{"qty": i.get("qty"), "name": i.get("name")} for i in o.get("items", [])],
            "time": o.get("created_at").strftime("%H:%M") if isinstance(o.get("created_at"), datetime) else ""
        })
    stats = {
        "today_orders": today_orders,
        "today_revenue": money(today_revenue_cents),
        "month_orders": month_orders,
        "month_revenue": money(month_revenue_cents)
    }
    return render_template("admin.html", stats=stats, orders=orders_data, users=users_data)


@app.post("/admin/users")
def admin_create_user() -> Response:
    if "admin_user" not in session:
        return redirect(url_for("admin_login"))
        
    username = request.form.get("username")
    password = request.form.get("password")
    shift = request.form.get("shift") or "Morning"
    
    if not username or not password:
         return redirect(url_for("admin_dashboard")) # simplified error handling
         
    users_coll = db["users"]
    existing = users_coll.find_one({"username": username})
    if existing:
        return redirect(url_for("admin_dashboard"))
    next_id = db["counters"].find_one_and_update(
        {"_id": "users"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )["seq"]
    users_coll.insert_one({"id": next_id, "username": username, "password": password, "role": "kitchen", "shift": shift})
        


@app.post("/admin/users/delete")
def admin_delete_user() -> Response:
    if "admin_user" not in session:
        return redirect(url_for("admin_login"))
        
    user_id = request.form.get("user_id")
    if user_id:
        db["users"].delete_one({"id": int(user_id)})
                


@app.get("/admin/logout")
def admin_logout() -> Response:
    session.pop("admin_user", None)
    return redirect(url_for("admin_login"))


@app.delete("/api/orders/<int:order_id>")
def api_delete_order(order_id: int) -> Response:
    if "admin_user" not in session:
        return Response("Unauthorized", status=401)
        
    o = db["orders"].find_one({"id": order_id})
    if o is None:
        return Response("not found", status=404)
    db["orders"].delete_one({"id": order_id})
    return jsonify({"status": "deleted", "id": order_id})


@app.put("/api/orders/<int:order_id>")
def api_update_order(order_id: int) -> Response:
    if "admin_user" not in session:
        return Response("Unauthorized", status=401)
        
    data = request.get_json(force=True, silent=True) or {}
    status = data.get("status")
    
    order = db["orders"].find_one({"id": order_id})
    if order is None:
        return Response("not found", status=404)
    updates: Dict[str, Any] = {}
    if status:
        updates["status"] = status
        kt = order.get("kitchen_ticket")
        if kt:
            if status in ["preparing", "ready", "completed"]:
                updates["kitchen_ticket.status"] = status
            elif status == "paid":
                updates["kitchen_ticket.status"] = "new"
        if status == "paid":
            subtotal, tax, total = order_totals(order)
            pay = order.get("payment") or {}
            pay.update({
                "provider": pay.get("provider", "admin"),
                "status": "paid",
                "amount_cents": total,
                "reference": pay.get("reference", f"admin-{int(time.time())}"),
                "created_at": utcnow()
            })
            updates["payment"] = pay
    updates["updated_at"] = utcnow()
    db["orders"].update_one({"id": order_id}, {"$set": updates})
    return jsonify({"status": "updated", "id": order_id, "new_status": updates.get("status", order.get("status"))})


@app.route("/kitchen/login", methods=["GET", "POST"])
def kitchen_login() -> Response:
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = db["users"].find_one({"username": username})
        if user and user.get("password") == password and user.get("role") == "kitchen":
            session["kitchen_user"] = username
            session["kitchen_shift"] = request.form.get("shift") or user.get("shift", "Morning")
            return redirect(url_for("kitchen"))
        else:
            return render_template("kitchen_login.html", error="Invalid credentials")
    
    return render_template("kitchen_login.html")


@app.get("/kitchen/logout")
def kitchen_logout() -> Response:
    session.pop("kitchen_user", None)
    session.pop("kitchen_shift", None)
    return redirect(url_for("kitchen_login"))


@app.get("/kitchen")
def kitchen() -> Response:
    if not session.get("kitchen_user"):
        return redirect(url_for("kitchen_login"))

    return render_template(
        "kitchen.html",
        user=session["kitchen_user"],
        shift=session.get("kitchen_shift", "Morning"),
    )


@app.get("/kitchen/menu")
def kitchen_menu() -> Response:
    if not session.get("kitchen_user"):
        return redirect(url_for("kitchen_login"))

    items = list(db["menu_items"].find({}, sort=[("category", 1), ("name", 1)]))
    menu_items = []
    for it in items:
        menu_items.append({
            "id": it.get("id"),
            "name": it.get("name"),
            "category": it.get("category"),
            "price_cents": it.get("price_cents"),
            "gst_percent": it.get("gst_percent"),
            "available": it.get("available"),
        })

    return render_template(
        "kitchen_menu.html",
        user=session["kitchen_user"],
        shift=session.get("kitchen_shift", "Morning"),
        menu_items=menu_items
    )


@app.post("/kitchen/menu/add")
def kitchen_menu_add() -> Response:
    if not session.get("kitchen_user"):
        return redirect(url_for("kitchen_login"))
        
    name = request.form.get("name")
    category = request.form.get("category")
    price_str = request.form.get("price") # Assume input is in rupees/dollars (float)
    gst_str = request.form.get("gst")
    
    if name and category and price_str and gst_str:
        try:
            price_cents = int(float(price_str) * 100)
            gst_percent = int(gst_str)
            
            existing = db["menu_items"].find_one({"name": name})
            if existing:
                return redirect(url_for("kitchen_menu"))
            next_id = db["counters"].find_one_and_update(
                {"_id": "menu_items"},
                {"$inc": {"seq": 1}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )["seq"]
            db["menu_items"].insert_one({
                "id": next_id,
                "name": name,
                "category": category,
                "price_cents": price_cents,
                "gst_percent": gst_percent,
                "available": True
            })
        except ValueError:
            pass # Handle error
            
    return redirect(url_for("kitchen_menu"))


@app.post("/kitchen/menu/edit")
def kitchen_menu_edit() -> Response:
    if not session.get("kitchen_user"):
        return redirect(url_for("kitchen_login"))
        
    item_id = request.form.get("id")
    name = request.form.get("name")
    category = request.form.get("category")
    price_str = request.form.get("price")
    gst_str = request.form.get("gst")
    available = request.form.get("available") == "on"
    
    if item_id:
        updates: Dict[str, Any] = {}
        if name: updates["name"] = name
        if category: updates["category"] = category
        if price_str:
            try:
                updates["price_cents"] = int(float(price_str) * 100)
            except ValueError:
                pass
        if gst_str:
            try:
                updates["gst_percent"] = int(gst_str)
            except ValueError:
                pass
        updates["available"] = available
        db["menu_items"].update_one({"id": int(item_id)}, {"$set": updates})
                
    return redirect(url_for("kitchen_menu"))


@app.post("/kitchen/menu/delete")
def kitchen_menu_delete() -> Response:
    if not session.get("kitchen_user"):
        return redirect(url_for("kitchen_login"))
        
    item_id = request.form.get("id")
    if item_id:
        db["menu_items"].delete_one({"id": int(item_id)})
                


@app.post("/kitchen/menu/availability")
def kitchen_menu_availability() -> Response:
    if not session.get("kitchen_user"):
        return redirect(url_for("kitchen_login"))
    visible_ids = [int(x) for x in request.form.getlist("visible_ids")]
    all_ids = [int(d.get("id")) for d in db["menu_items"].find({}, projection={"id": 1})]
    db["menu_items"].update_many({"id": {"$in": visible_ids}}, {"$set": {"available": True}})
    db["menu_items"].update_many({"id": {"$nin": visible_ids}}, {"$set": {"available": False}})
    return redirect(url_for("kitchen_menu"))


@app.get("/api/menu")
def api_menu() -> Response:
    items = list(db["menu_items"].find({}, sort=[("category", 1), ("name", 1)]))
    return jsonify(
        [
            {
                "id": it.get("id"),
                "name": it.get("name"),
                "category": it.get("category"),
                "price_cents": it.get("price_cents"),
                "gst_percent": it.get("gst_percent"),
                "available": it.get("available"),
            }
            for it in items
        ]
    )


@app.post("/api/nlu")
def api_nlu() -> Response:
    data = request.get_json(force=True, silent=True) or {}
    transcript = str(data.get("transcript") or "")
    menu_docs = list(db["menu_items"].find({"available": True}))
    menu = [MenuLike(d) for d in menu_docs]
    items = nlu_extract_order(transcript, menu)
    return jsonify({"items": items})


@app.post("/api/orders")
def api_create_order() -> Response:
    data = request.get_json(force=True, silent=True) or {}
    table_id = str(data.get("table_id") or "T1")
    items = list(data.get("items") or [])
    now = utcnow()
    cid = session.get("customer_id")
    cust = db["customers"].find_one({"id": int(cid)}) if cid else None

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    max_daily = db["orders"].find_one(
        {"created_at": {"$gte": today_start}},
        sort=[("daily_id", -1)],
        projection={"daily_id": 1},
    )
    daily_id = int((max_daily or {}).get("daily_id", 0)) + 1
    next_id = db["counters"].find_one_and_update(
        {"_id": "orders"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )["seq"]
    order_items: List[Dict[str, Any]] = []
    for it in items:
        menu_item_id = int(it.get("menu_item_id"))
        qty = int(it.get("qty") or 1)
        notes = str(it.get("notes") or "")
        menu_item = db["menu_items"].find_one({"id": menu_item_id, "available": True})
        if not menu_item:
            continue
        order_items.append({
            "menu_item_id": menu_item["id"],
            "name": menu_item["name"],
            "price_cents": int(menu_item["price_cents"]),
            "gst_percent": int(menu_item.get("gst_percent", 5)),
            "qty": max(1, qty),
            "notes": notes,
        })
    order_doc = {
        "id": next_id,
        "daily_id": daily_id,
        "table_id": table_id,
        "status": "open",
        "created_at": now,
        "updated_at": now,
        "items": order_items,
        "customer": (
            {"id": int(cust.get("id")), "name": cust.get("name"), "phone": cust.get("phone")}
            if cust else None
        ),
    }
    db["orders"].insert_one(order_doc)
    return jsonify(
        {
            "id": order_doc["id"],
            "daily_id": order_doc["daily_id"] or order_doc["id"],
            "table_id": order_doc["table_id"],
            "status": order_doc["status"],
        }
    )


@app.get("/api/orders/<int:order_id>")
def api_get_order(order_id: int) -> Response:
    order = db["orders"].find_one({"id": order_id})
    if order is None:
        return Response("not found", status=404)
    subtotal, tax, total = order_totals(order)
    return jsonify(
        {
            "id": order.get("id"),
            "daily_id": order.get("daily_id") or order.get("id"),
            "table_id": order.get("table_id"),
            "status": order.get("status"),
            "subtotal_cents": subtotal,
            "tax_cents": tax,
            "total_cents": total,
            "items": [
                {
                    "menu_item_id": it.get("menu_item_id"),
                    "name": it.get("name"),
                    "qty": it.get("qty"),
                    "notes": it.get("notes"),
                }
                for it in order.get("items", [])
            ],
            "payment": order.get("payment"),
            "kitchen_ticket": order.get("kitchen_ticket"),
        }
    )


@app.post("/api/orders/<int:order_id>/checkout")
def api_checkout(order_id: int) -> Response:
    order = db["orders"].find_one({"id": order_id})
    if order is None:
        return Response("not found", status=404)
    subtotal, tax, total = order_totals(order)
    now = utcnow()
    pay = order.get("payment") or {}
    pay.update({
        "provider": "demo",
        "status": "pending",
        "amount_cents": total,
        "reference": f"demo-{order_id}-{int(time.time())}",
        "created_at": now,
    })
    db["orders"].update_one({"id": order_id}, {"$set": {"payment": pay, "status": "awaiting_payment", "updated_at": now}})

    pay_url = f"/pay/{order_id}"
    return jsonify({"order_id": order_id, "total_cents": total, "pay_url": pay_url})


@app.get("/pay/<int:order_id>")
def pay_page(order_id: int) -> Response:
    order = db["orders"].find_one({"id": order_id})
    if order is None:
        return Response("not found", status=404)
    subtotal, tax, total = order_totals(order)
    status = (order.get("payment") or {}).get("status", "pending")
    items_list = []
    tax_breakdown: Dict[int, int] = {}
    for it in order.get("items", []):
        price = int(it.get("price_cents", 0))
        qty = int(it.get("qty", 1))
        gst = int(it.get("gst_percent", 0))
        item_subtotal = price * qty
        item_tax = int(round(item_subtotal * gst / 100.0))
        tax_breakdown[gst] = tax_breakdown.get(gst, 0) + item_tax
        items_list.append({
            "name": it.get("name"),
            "qty": qty,
            "price_fmt": money(price),
        })
    taxes_list = []
    for rate in sorted(tax_breakdown.keys()):
        taxes_list.append({
            "rate": rate,
            "amount_fmt": money(tax_breakdown[rate])
        })

    pay_link = f"{request.url_root.rstrip('/')}/pay/{order_id}"
    return render_template(
        "pay.html",
        order_id=order_id,
        daily_id=order.daily_id or order_id,
        status=status,
        subtotal_fmt=money(subtotal),
        taxes_list=taxes_list,
        total_fmt=money(total),
        pay_link=pay_link,
        items=items_list,
    )


@app.post("/api/payments/<int:order_id>/complete")
def api_payment_complete(order_id: int) -> Response:
    order = db["orders"].find_one({"id": order_id})
    if order is None:
        return Response("not found", status=404)
    subtotal, tax, total = order_totals(order)
    now = utcnow()
    pay = order.get("payment") or {}
    pay.update({
        "provider": "demo",
        "status": "paid",
        "amount_cents": total,
        "reference": f"demo-{order_id}-{int(time.time())}",
        "created_at": now,
    })
    kt = order.get("kitchen_ticket") or {}
    kt.update({"status": "new", "updated_at": now})
    db["orders"].update_one({"id": order_id}, {"$set": {"payment": pay, "kitchen_ticket": kt, "status": "paid", "updated_at": now}})

    cust = order.get("customer") or {}
    redirect_url = f"/customer/{int(cust.get('id'))}" if cust.get("id") is not None else None
    return jsonify({"order_id": order_id, "status": "paid", "customer_id": cust.get("id"), "redirect_url": redirect_url})


@app.get("/api/kitchen/tickets")
def api_kitchen_tickets() -> Response:
    orders = list(db["orders"].find({"status": {"$in": ["paid", "preparing", "ready"]}}, sort=[("id", -1)]))
    tickets = []
    for o in orders:
        kt = o.get("kitchen_ticket") or {}
        kitchen_status = kt.get("status", "new")
        tickets.append(
            {
                "order_id": o.get("id"),
                "daily_id": o.get("daily_id") or o.get("id"),
                "table_id": o.get("table_id"),
                "kitchen_status": kitchen_status,
                "customer_name": (o.get("customer") or {}).get("name"),
                "customer_phone": (o.get("customer") or {}).get("phone"),
                "items": [
                    {"name": it.get("name"), "qty": it.get("qty"), "notes": it.get("notes")}
                    for it in o.get("items", [])
                ],
            }
        )
    return jsonify({"tickets": tickets})


@app.post("/api/kitchen/<int:order_id>/status")
def api_kitchen_status(order_id: int) -> Response:
    data = request.get_json(force=True, silent=True) or {}
    status = str(data.get("status") or "").strip().lower()
    if status not in {"new", "preparing", "ready", "completed"}:
        return Response("invalid status", status=400)

    order = db["orders"].find_one({"id": order_id})
    if order is None:
        return Response("not found", status=404)
    now = utcnow()
    kt = order.get("kitchen_ticket") or {}
    kt.update({"status": status, "updated_at": now})
    new_status = status if status in {"preparing", "ready", "completed"} else order.get("status")
    db["orders"].update_one({"id": order_id}, {"$set": {"kitchen_ticket": kt, "status": new_status, "updated_at": now}})
    return jsonify({"order_id": order_id, "kitchen_status": status})


TARGET_SR = 16_000


def resample_audio(
    x: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR
) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return x.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // g
    down = int(orig_sr) // g
    y = resample_poly(x, up, down).astype(np.float32, copy=False)
    return y


@dataclass(frozen=True)
class ManifestRow:
    audio_filepath: str
    sampling_rate: int
    text: str


def load_manifest(path: Path, limit: int = 2000) -> List[ManifestRow]:
    if not path.exists():
        return []
    rows: List[ManifestRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= limit:
                break
            obj = json.loads(line)
            rows.append(
                ManifestRow(
                    audio_filepath=str(obj.get("audio_filepath") or ""),
                    sampling_rate=int(obj.get("sampling_rate") or 0),
                    text=str(obj.get("text") or ""),
                )
            )
    return rows


MINDS_MANIFEST = DATA_DIR / "minds14" / "en-US" / "train" / "manifest.jsonl"
SLURP_MANIFEST = DATA_DIR / "slurp" / "train" / "manifest.jsonl"

MANIFESTS: Dict[str, List[ManifestRow]] = {
    "minds14": load_manifest(MINDS_MANIFEST, limit=2000),
    "slurp": load_manifest(SLURP_MANIFEST, limit=2000),
}


def safe_dataset_audio_path(dataset: str, index: int) -> Optional[Path]:
    rows = MANIFESTS.get(dataset)
    if not rows:
        return None
    if index < 0 or index >= len(rows):
        return None
    p = Path(rows[index].audio_filepath)
    try:
        rp = p.resolve()
        if DATA_DIR.resolve() not in rp.parents and rp != DATA_DIR.resolve():
            return None
        return rp
    except Exception:
        return None


@app.get("/api/datasets")
def api_datasets() -> Response:
    info = {}
    for k, rows in MANIFESTS.items():
        info[k] = {"examples": len(rows)}
    return jsonify(info)


@app.get("/api/datasets/<dataset>/sample")
def api_dataset_sample(dataset: str) -> Response:
    idx = int(request.args.get("index") or 0)
    rows = MANIFESTS.get(dataset)
    if not rows:
        return Response("unknown dataset", status=404)
    if idx < 0 or idx >= len(rows):
        return Response("index out of range", status=400)
    row = rows[idx]
    return jsonify(
        {
            "dataset": dataset,
            "index": idx,
            "text": row.text,
            "audio_url": f"/dataset_audio/{dataset}/{idx}.wav",
            "sampling_rate": row.sampling_rate,
        }
    )


@app.get("/dataset_audio/<dataset>/<path:filename>")
def dataset_audio(dataset: str, filename: str) -> Response:
    m = re.fullmatch(r"(\d+)\.wav", filename)
    if not m:
        return Response("bad request", status=400)
    idx = int(m.group(1))
    p = safe_dataset_audio_path(dataset, idx)
    if p is None or not p.exists():
        return Response("not found", status=404)
    return send_file(
        str(p), mimetype="audio/wav", as_attachment=False, download_name=p.name
    )


_asr_processor = None
_asr_model = None
_asr_lock = threading.Lock()


def get_asr():
    global _asr_processor, _asr_model
    if _asr_processor is not None and _asr_model is not None:
        return _asr_processor, _asr_model

    with _asr_lock:
        if _asr_processor is not None and _asr_model is not None:
            return _asr_processor, _asr_model

        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except Exception as e:
            raise RuntimeError(f"transformers import failed: {e}") from e

        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()

        _asr_processor = processor
        _asr_model = model
        return processor, model


def transcribe_with_asr(audio_16k: np.ndarray) -> str:
    processor, model = get_asr()
    inputs = processor(
        audio_16k, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    import torch

    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(predicted_ids)[0]
    return transcript


@app.get("/api/asr_sample")
def api_asr_sample() -> Response:
    dataset = str(request.args.get("dataset") or "minds14")
    idx = int(request.args.get("index") or 0)

    rows = MANIFESTS.get(dataset)
    if not rows:
        return Response("unknown dataset", status=404)
    if idx < 0 or idx >= len(rows):
        return Response("index out of range", status=400)

    p = safe_dataset_audio_path(dataset, idx)
    if p is None or not p.exists():
        return Response("audio not found", status=404)

    try:
        audio, sr = sf.read(str(p), always_2d=False)
    except Exception:
        return Response("could not decode audio", status=400)

    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != TARGET_SR:
        audio = resample_audio(audio, orig_sr=int(sr), target_sr=TARGET_SR)

    try:
        transcript = transcribe_with_asr(audio)
    except Exception as e:
        return Response(str(e), status=500)

    return jsonify(
        {
            "dataset": dataset,
            "index": idx,
            "ground_truth": rows[idx].text,
            "transcript": transcript,
            "sampling_rate": TARGET_SR,
        }
    )


@app.post("/api/asr")
def api_asr() -> Response:
    if "audio" not in request.files:
        return Response("missing file field 'audio'", status=400)
    f = request.files["audio"]
    raw = f.read()
    if not raw:
        return Response("empty file", status=400)

    try:
        audio, sr = sf.read(io.BytesIO(raw), always_2d=False)
    except Exception:
        return Response("could not decode audio (send WAV/FLAC)", status=400)

    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != TARGET_SR:
        audio = resample_audio(audio, orig_sr=int(sr), target_sr=TARGET_SR)
    try:
        transcript = transcribe_with_asr(audio)
    except Exception as e:
        return Response(str(e), status=500)
    return jsonify({"transcript": transcript, "sampling_rate": TARGET_SR})





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
