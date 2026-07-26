// 性能计时器模块
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// 单个计时阶段的状态。
struct Stage {
    start: Instant,
    elapsed: Duration,
}

/// 性能计时器。
/// 使用单一 Mutex 保护所有状态，避免双锁竞争。
pub struct PerfTimer {
    stages: Mutex<HashMap<String, Stage>>,
    enabled: bool,
}

impl PerfTimer {
    pub fn new(enabled: bool) -> Self {
        Self {
            stages: Mutex::new(HashMap::new()),
            enabled,
        }
    }

    pub fn start(&self, name: &str) {
        if !self.enabled {
            return;
        }
        let mut stages = self.stages.lock().unwrap();
        stages.insert(
            name.to_string(),
            Stage {
                start: Instant::now(),
                elapsed: Duration::ZERO,
            },
        );
    }

    pub fn stop(&self, name: &str) {
        if !self.enabled {
            return;
        }
        let mut stages = self.stages.lock().unwrap();
        if let Some(stage) = stages.get_mut(name) {
            stage.elapsed = stage.start.elapsed();
        }
    }

    pub fn elapsed_ms(&self, name: &str) -> i64 {
        let stages = self.stages.lock().unwrap();
        stages
            .get(name)
            .map(|s| s.elapsed.as_millis() as i64)
            .unwrap_or(0)
    }
}
