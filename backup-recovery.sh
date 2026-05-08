#!/bin/bash

# ZipIt MLOps Backup & Recovery System

BACKUP_DIR="/backups"
S3_BUCKET="zipit-mlops-backups"
RETENTION_DAYS=30

# Database backup
backup_database() {
    echo "Starting database backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # PostgreSQL backup
    docker exec postgres pg_dump -U zipit zipit_mlops > "$BACKUP_DIR/db_backup_$timestamp.sql"
    
    # Compress backup
    gzip "$BACKUP_DIR/db_backup_$timestamp.sql"
    
    echo "Database backup completed: db_backup_$timestamp.sql.gz"
}

# Application data backup
backup_app_data() {
    echo "Starting application data backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Backup uploaded models and files
    tar -czf "$BACKUP_DIR/app_data_$timestamp.tar.gz" ./models ./static/uploads
    
    echo "Application data backup completed: app_data_$timestamp.tar.gz"
}

# Configuration backup
backup_config() {
    echo "Starting configuration backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Backup configuration files
    tar -czf "$BACKUP_DIR/config_$timestamp.tar.gz" \
        ./docker-compose-production.yml \
        ./nginx-production.conf \
        ./monitoring/ \
        ./.env
    
    echo "Configuration backup completed: config_$timestamp.tar.gz"
}

# Upload to S3
upload_to_s3() {
    echo "Uploading backups to S3..."
    
    if command -v aws &> /dev/null; then
        aws s3 sync $BACKUP_DIR s3://$S3_BUCKET/$(date +%Y/%m/%d)/
        echo "Backups uploaded to S3"
    else
        echo "AWS CLI not found, skipping S3 upload"
    fi
}

# Clean old backups
cleanup_old_backups() {
    echo "Cleaning up old backups..."
    find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete
    find $BACKUP_DIR -name "*.sql" -mtime +$RETENTION_DAYS -delete
    echo "Old backups cleaned up"
}

# Health check and monitoring
health_check() {
    echo "Performing health checks..."
    
    # Check application health
    for i in {1..3}; do
        if curl -f http://zipit-app-$i:8000/health > /dev/null 2>&1; then
            echo "✅ App instance $i is healthy"
        else
            echo "❌ App instance $i is unhealthy"
            # Send alert
            send_alert "App instance $i is down"
        fi
    done
    
    # Check database health
    if docker exec postgres pg_isready -U zipit > /dev/null 2>&1; then
        echo "✅ Database is healthy"
    else
        echo "❌ Database is unhealthy"
        send_alert "Database is down"
    fi
    
    # Check nginx health
    if curl -f http://nginx/nginx-health > /dev/null 2>&1; then
        echo "✅ Load balancer is healthy"
    else
        echo "❌ Load balancer is unhealthy"
        send_alert "Load balancer is down"
    fi
}

# Send alert function
send_alert() {
    local message=$1
    echo "ALERT: $message"
    
    # Send email alert if configured
    if [ ! -z "$ALERT_EMAIL" ]; then
        echo "$message" | mail -s "ZipIt MLOps Alert" $ALERT_EMAIL
    fi
    
    # Send to Slack if webhook configured
    if [ ! -z "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"ZipIt MLOps Alert: $message\"}" \
            $SLACK_WEBHOOK
    fi
}

# Disaster recovery
disaster_recovery() {
    echo "Starting disaster recovery..."
    
    # Stop all services
    docker-compose -f docker-compose-production.yml down
    
    # Restore from latest backup
    latest_db_backup=$(ls -t $BACKUP_DIR/db_backup_*.sql.gz | head -1)
    latest_app_backup=$(ls -t $BACKUP_DIR/app_data_*.tar.gz | head -1)
    
    if [ -f "$latest_db_backup" ]; then
        echo "Restoring database from $latest_db_backup"
        gunzip -c "$latest_db_backup" | docker exec -i postgres psql -U zipit zipit_mlops
    fi
    
    if [ -f "$latest_app_backup" ]; then
        echo "Restoring application data from $latest_app_backup"
        tar -xzf "$latest_app_backup"
    fi
    
    # Restart services
    docker-compose -f docker-compose-production.yml up -d
    
    echo "Disaster recovery completed"
}

# Main execution
case "$1" in
    "backup")
        mkdir -p $BACKUP_DIR
        backup_database
        backup_app_data
        backup_config
        upload_to_s3
        cleanup_old_backups
        ;;
    "health")
        health_check
        ;;
    "recover")
        disaster_recovery
        ;;
    "monitor")
        while true; do
            health_check
            sleep 300  # Check every 5 minutes
        done
        ;;
    *)
        echo "Usage: $0 {backup|health|recover|monitor}"
        exit 1
        ;;
esac