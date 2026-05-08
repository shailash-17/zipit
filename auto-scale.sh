#!/bin/bash

# ZipIt MLOps Auto-Scaling Script

DOCKER_COMPOSE_FILE="docker-compose-production.yml"
MIN_INSTANCES=2
MAX_INSTANCES=10
CPU_THRESHOLD=70
MEMORY_THRESHOLD=80

# Function to get current CPU usage
get_cpu_usage() {
    docker stats --no-stream --format "table {{.CPUPerc}}" | grep -v CPU | sed 's/%//' | awk '{sum+=$1} END {print sum/NR}'
}

# Function to get current memory usage
get_memory_usage() {
    docker stats --no-stream --format "table {{.MemPerc}}" | grep -v MEM | sed 's/%//' | awk '{sum+=$1} END {print sum/NR}'
}

# Function to get current number of app instances
get_instance_count() {
    docker-compose -f $DOCKER_COMPOSE_FILE ps -q | grep zipit-app | wc -l
}

# Function to scale up
scale_up() {
    current_instances=$(get_instance_count)
    if [ $current_instances -lt $MAX_INSTANCES ]; then
        new_count=$((current_instances + 1))
        echo "Scaling up to $new_count instances"
        
        # Add new instance to docker-compose
        docker-compose -f $DOCKER_COMPOSE_FILE up -d --scale zipit-app=$new_count
        
        # Update nginx upstream
        nginx -s reload
        
        echo "Scaled up successfully"
    else
        echo "Already at maximum instances ($MAX_INSTANCES)"
    fi
}

# Function to scale down
scale_down() {
    current_instances=$(get_instance_count)
    if [ $current_instances -gt $MIN_INSTANCES ]; then
        new_count=$((current_instances - 1))
        echo "Scaling down to $new_count instances"
        
        # Remove instance from docker-compose
        docker-compose -f $DOCKER_COMPOSE_FILE up -d --scale zipit-app=$new_count
        
        # Update nginx upstream
        nginx -s reload
        
        echo "Scaled down successfully"
    else
        echo "Already at minimum instances ($MIN_INSTANCES)"
    fi
}

# Main monitoring loop
while true; do
    cpu_usage=$(get_cpu_usage)
    memory_usage=$(get_memory_usage)
    current_instances=$(get_instance_count)
    
    echo "$(date): CPU: ${cpu_usage}%, Memory: ${memory_usage}%, Instances: $current_instances"
    
    # Scale up if high usage
    if (( $(echo "$cpu_usage > $CPU_THRESHOLD" | bc -l) )) || (( $(echo "$memory_usage > $MEMORY_THRESHOLD" | bc -l) )); then
        echo "High resource usage detected, scaling up..."
        scale_up
    fi
    
    # Scale down if low usage
    if (( $(echo "$cpu_usage < 30" | bc -l) )) && (( $(echo "$memory_usage < 40" | bc -l) )) && [ $current_instances -gt $MIN_INSTANCES ]; then
        echo "Low resource usage detected, scaling down..."
        scale_down
    fi
    
    # Wait before next check
    sleep 60
done