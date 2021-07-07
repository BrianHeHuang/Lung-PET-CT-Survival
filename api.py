from flask import jsonify

from app import app
from db import Result


@app.route("/results")
def results():
    results = Result.query.all()
    return jsonify([r.dict() for r in results])
