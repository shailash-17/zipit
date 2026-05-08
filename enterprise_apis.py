# Enterprise API Endpoints - Add to mlops_platform.py

# Subscription & Payment APIs
@app.post("/api/subscription/upgrade")
async def upgrade_subscription(
    upgrade: SubscriptionUpgrade,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not stripe.api_key:
        raise HTTPException(status_code=400, detail="Payment system not configured")
    
    plan = SUBSCRIPTION_PLANS.get(upgrade.plan)
    if not plan:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    try:
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.full_name
            )
            current_user.stripe_customer_id = customer.id
        
        subscription = stripe.Subscription.create(
            customer=current_user.stripe_customer_id,
            items=[{"price": f"price_{upgrade.plan}"}],
            metadata={"user_id": current_user.id}
        )
        
        current_user.subscription_tier = upgrade.plan
        current_user.subscription_status = "active"
        current_user.subscription_end_date = datetime.fromtimestamp(subscription.current_period_end)
        
        db_subscription = Subscription(
            user_id=current_user.id,
            plan=upgrade.plan,
            status="active",
            stripe_subscription_id=subscription.id,
            current_period_start=datetime.fromtimestamp(subscription.current_period_start),
            current_period_end=datetime.fromtimestamp(subscription.current_period_end)
        )
        db.add(db_subscription)
        db.commit()
        
        log_audit(db, current_user.id, "upgrade_subscription", f"plan_{upgrade.plan}", {"plan": upgrade.plan})
        
        return {"message": "Subscription upgraded successfully", "plan": upgrade.plan}
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/subscription/status")
async def get_subscription_status(current_user: User = Depends(get_current_user)):
    plan = SUBSCRIPTION_PLANS.get(current_user.subscription_tier, SUBSCRIPTION_PLANS["free-tier"])
    
    return {
        "plan": current_user.subscription_tier,
        "status": current_user.subscription_status,
        "limits": {
            "models": plan["models"],
            "predictions": plan["predictions"],
            "features": plan["features"]
        },
        "usage": {
            "models": current_user.monthly_models,
            "predictions": current_user.monthly_predictions
        },
        "end_date": current_user.subscription_end_date
    }

# A/B Testing APIs
@app.post("/api/models/{model_id}/ab-test")
async def create_ab_test(
    model_id: int,
    ab_test: ABTestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(MLModel.id == model_id, MLModel.user_id == current_user.id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    plan = SUBSCRIPTION_PLANS.get(current_user.subscription_tier, SUBSCRIPTION_PLANS["free-tier"])
    if "ab_testing" not in plan.get("features", []) and "all_features" not in plan.get("features", []):
        raise HTTPException(status_code=403, detail="A/B testing not available in your plan")
    
    db_ab_test = ABTest(
        model_id=model_id,
        name=ab_test.name,
        description=ab_test.description,
        control_version=ab_test.control_version,
        treatment_version=ab_test.treatment_version,
        traffic_split=ab_test.traffic_split
    )
    db.add(db_ab_test)
    db.commit()
    
    log_audit(db, current_user.id, "create_ab_test", f"model_{model_id}", {"test_name": ab_test.name})
    
    return {"message": "A/B test created successfully", "test_id": db_ab_test.id}

@app.get("/api/models/{model_id}/ab-tests")
async def get_ab_tests(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    model = db.query(MLModel).filter(MLModel.id == model_id, MLModel.user_id == current_user.id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    tests = db.query(ABTest).filter(ABTest.model_id == model_id).all()
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "status": t.status,
            "traffic_split": t.traffic_split,
            "start_date": t.start_date,
            "results": t.results
        } for t in tests
    ]

# Advanced Analytics APIs
@app.get("/api/analytics/business-impact")
async def get_business_impact(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    models = db.query(MLModel).filter(MLModel.user_id == current_user.id).all()
    
    total_predictions = sum(m.total_predictions for m in models)
    avg_accuracy = sum(m.accuracy for m in models) / len(models) if models else 0
    
    estimated_value = total_predictions * 0.10
    
    return {
        "total_models": len(models),
        "total_predictions": total_predictions,
        "average_accuracy": round(avg_accuracy * 100, 2),
        "estimated_business_value": round(estimated_value, 2),
        "cost_savings": round(estimated_value * 0.3, 2),
        "roi_percentage": 250
    }

# Organizations APIs
@app.post("/api/organizations")
async def create_organization(
    name: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    existing = db.query(Organization).filter(Organization.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Organization already exists")
    
    org = Organization(name=name)
    db.add(org)
    db.commit()
    db.refresh(org)
    
    current_user.organization_id = org.id
    current_user.role = "owner"
    db.commit()
    
    log_audit(db, current_user.id, "create_organization", f"org_{org.id}", {"name": name})
    
    return {"message": "Organization created successfully", "organization_id": org.id}

# Audit & Compliance APIs
@app.get("/api/audit/logs")
async def get_audit_logs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "owner"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    logs = db.query(AuditLog).filter(AuditLog.user_id == current_user.id).order_by(AuditLog.timestamp.desc()).limit(100).all()
    
    return [
        {
            "action": log.action,
            "resource": log.resource,
            "details": log.details,
            "timestamp": log.timestamp,
            "ip_address": log.ip_address
        } for log in logs
    ]

# Enhanced Model APIs with usage tracking
@app.post("/api/models/register")
async def register_model(
    model_name: str = Form(...),
    model_type: str = Form(...),
    framework: str = Form(default="sklearn"),
    description: str = Form(default=""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not check_usage_limits(current_user, "model_creation"):
        raise HTTPException(status_code=403, detail="Model limit exceeded for your plan")
    
    existing = db.query(MLModel).filter(MLModel.user_id == current_user.id, MLModel.model_name == model_name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already exists")
    
    ml_model, accuracy = create_sample_model(model_type)
    model_data = pickle.dumps(ml_model).hex()
    
    db_model = MLModel(
        user_id=current_user.id,
        model_name=model_name,
        model_type=model_type,
        framework=framework,
        description=description,
        model_data=model_data,
        accuracy=accuracy
    )
    db.add(db_model)
    
    current_user.monthly_models += 1
    db.commit()
    
    log_audit(db, current_user.id, "register_model", f"model_{db_model.id}", {"model_name": model_name})
    
    return {"message": "Model registered successfully", "model_id": db_model.id, "accuracy": round(accuracy * 100, 2)}

# Monitoring & Alerting APIs
@app.post("/api/alerts")
async def create_alert(
    model_id: int = Form(...),
    alert_type: str = Form(...),
    severity: str = Form(...),
    message: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = Alert(
        user_id=current_user.id,
        model_id=model_id,
        alert_type=alert_type,
        severity=severity,
        message=message
    )
    db.add(alert)
    db.commit()
    
    # Send email notification if enabled
    if EMAIL_ENABLED and severity in ["high", "critical"]:
        send_email(
            current_user.email,
            f"ZipIt Alert: {alert_type}",
            f"<h2>Alert: {alert_type}</h2><p>{message}</p><p>Severity: {severity}</p>"
        )
    
    return {"message": "Alert created successfully", "alert_id": alert.id}

@app.get("/api/alerts")
async def get_alerts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    alerts = db.query(Alert).filter(Alert.user_id == current_user.id, Alert.status == "active").order_by(Alert.created_at.desc()).limit(50).all()
    
    return [
        {
            "id": a.id,
            "alert_type": a.alert_type,
            "severity": a.severity,
            "message": a.message,
            "created_at": a.created_at,
            "status": a.status
        } for a in alerts
    ]