/**
 * Real-time Dashboard and Monitoring
 * Features: Live metrics, system health, prediction history
 */

class Dashboard {
    constructor() {
        this.metrics = {
            totalPredictions: 1247,
            modelAccuracy: 82.3,
            avgProcessingTime: 0.45,
            systemStatus: 'healthy'
        };
        
        this.recentPredictions = [];
        this.updateInterval = null;
        
        this.initializeDashboard();
        this.startRealTimeUpdates();
    }

    initializeDashboard() {
        this.updateMetrics();
        this.loadRecentPredictions();
        this.setupHealthChecks();
    }

    updateMetrics() {
        document.getElementById('modelAccuracy').textContent = `${this.metrics.modelAccuracy}%`;
        document.getElementById('totalPredictions').textContent = this.metrics.totalPredictions.toLocaleString();
        document.getElementById('avgProcessingTime').textContent = `${this.metrics.avgProcessingTime}s`;
        
        const statusElement = document.getElementById('systemStatus');
        statusElement.textContent = this.getStatusEmoji() + ' ' + this.metrics.systemStatus.charAt(0).toUpperCase() + this.metrics.systemStatus.slice(1);
        statusElement.className = `metric-value status-${this.metrics.systemStatus}`;
    }

    getStatusEmoji() {
        switch (this.metrics.systemStatus) {
            case 'healthy': return '🟢';
            case 'warning': return '🟡';
            case 'error': return '🔴';
            default: return '⚪';
        }
    }

    startRealTimeUpdates() {
        // Update metrics every 30 seconds
        this.updateInterval = setInterval(() => {
            this.simulateMetricUpdates();
            this.updateMetrics();
        }, 30000);
    }

    simulateMetricUpdates() {
        // Simulate realistic metric changes
        this.metrics.totalPredictions += Math.floor(Math.random() * 5);
        this.metrics.modelAccuracy += (Math.random() - 0.5) * 0.1;
        this.metrics.avgProcessingTime += (Math.random() - 0.5) * 0.05;
        
        // Keep metrics within realistic bounds
        this.metrics.modelAccuracy = Math.max(75, Math.min(90, this.metrics.modelAccuracy));
        this.metrics.avgProcessingTime = Math.max(0.2, Math.min(1.0, this.metrics.avgProcessingTime));
        
        // Random system status changes (rare)
        if (Math.random() < 0.01) {
            const statuses = ['healthy', 'warning', 'error'];
            this.metrics.systemStatus = statuses[Math.floor(Math.random() * statuses.length)];
        }
    }

    loadRecentPredictions() {
        const samplePredictions = [
            { keyword: 'yes', confidence: '87%', timestamp: new Date(Date.now() - 2 * 60 * 1000) },
            { keyword: 'no', confidence: '92%', timestamp: new Date(Date.now() - 5 * 60 * 1000) },
            { keyword: 'stop', confidence: '78%', timestamp: new Date(Date.now() - 8 * 60 * 1000) },
            { keyword: 'go', confidence: '85%', timestamp: new Date(Date.now() - 12 * 60 * 1000) },
            { keyword: 'up', confidence: '90%', timestamp: new Date(Date.now() - 15 * 60 * 1000) }
        ];
        
        this.recentPredictions = samplePredictions;
        this.renderRecentPredictions();
    }

    renderRecentPredictions() {
        const container = document.getElementById('recentPredictions');
        container.innerHTML = '';
        
        this.recentPredictions.forEach(prediction => {
            const predictionDiv = document.createElement('div');
            predictionDiv.className = 'prediction-item';
            predictionDiv.innerHTML = `
                <div>
                    <strong>${prediction.keyword}</strong>
                    <span class="confidence">${prediction.confidence}</span>
                </div>
                <span class="timestamp">${prediction.timestamp.toLocaleTimeString()}</span>
            `;
            container.appendChild(predictionDiv);
        });
    }

    addPrediction(keyword, confidence) {
        const newPrediction = {
            keyword: keyword,
            confidence: confidence,
            timestamp: new Date()
        };
        
        this.recentPredictions.unshift(newPrediction);
        
        // Keep only last 10 predictions
        if (this.recentPredictions.length > 10) {
            this.recentPredictions = this.recentPredictions.slice(0, 10);
        }
        
        this.renderRecentPredictions();
        this.metrics.totalPredictions++;
    }

    setupHealthChecks() {
        // Simulate health check every 10 seconds
        setInterval(() => {
            this.performHealthCheck();
        }, 10000);
    }

    async performHealthCheck() {
        try {
            // Simulate API health check
            const response = await fetch('/health', { method: 'GET' });
            
            if (response.ok) {
                this.metrics.systemStatus = 'healthy';
            } else {
                this.metrics.systemStatus = 'warning';
            }
        } catch (error) {
            this.metrics.systemStatus = 'error';
        }
        
        this.updateMetrics();
    }

    // Export metrics for external monitoring
    exportMetrics() {
        return {
            timestamp: new Date().toISOString(),
            metrics: this.metrics,
            recentPredictions: this.recentPredictions.slice(0, 5)
        };
    }

    // Cleanup when dashboard is destroyed
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Performance monitoring
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            responseTime: [],
            memoryUsage: [],
            errorRate: 0
        };
        
        this.startMonitoring();
    }

    startMonitoring() {
        // Monitor memory usage
        if (performance.memory) {
            setInterval(() => {
                this.metrics.memoryUsage.push({
                    timestamp: Date.now(),
                    used: performance.memory.usedJSHeapSize,
                    total: performance.memory.totalJSHeapSize
                });
                
                // Keep only last 100 measurements
                if (this.metrics.memoryUsage.length > 100) {
                    this.metrics.memoryUsage.shift();
                }
            }, 5000);
        }
        
        // Monitor page load performance
        window.addEventListener('load', () => {
            const navigation = performance.getEntriesByType('navigation')[0];
            if (navigation) {
                console.log('Page load time:', navigation.loadEventEnd - navigation.loadEventStart);
            }
        });
    }

    measureResponseTime(startTime) {
        const responseTime = Date.now() - startTime;
        this.metrics.responseTime.push(responseTime);
        
        // Keep only last 50 measurements
        if (this.metrics.responseTime.length > 50) {
            this.metrics.responseTime.shift();
        }
        
        return responseTime;
    }

    getAverageResponseTime() {
        if (this.metrics.responseTime.length === 0) return 0;
        return this.metrics.responseTime.reduce((a, b) => a + b, 0) / this.metrics.responseTime.length;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
    window.performanceMonitor = new PerformanceMonitor();
});

// Export for global access
window.Dashboard = Dashboard;
window.PerformanceMonitor = PerformanceMonitor;

