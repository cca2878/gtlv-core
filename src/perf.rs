//! 分阶段耗时计时器。
//!
//! `stop` 是**累加**语义：同名阶段可在循环里反复 start/stop，取到的是各段之和，
//! 这样逐个裁剪块计时也能汇总出总耗时。计时关闭时所有方法均为空操作。

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

struct Stage {
    start: Instant,
    elapsed: Duration,
}

/// 性能计时器。单一 Mutex 保护全部状态，避免双锁竞争。
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

    /// 开始（或继续）一个阶段的计时；已累计的耗时保留。
    pub fn start(&self, name: &str) {
        if !self.enabled {
            return;
        }
        // 计时不应影响求解，故锁中毒时静默跳过。
        let Ok(mut stages) = self.stages.lock() else {
            return;
        };
        stages
            .entry(name.to_string())
            .or_insert_with(|| Stage {
                start: Instant::now(),
                elapsed: Duration::ZERO,
            })
            .start = Instant::now();
    }

    /// 结束本段计时，把本段耗时累加进该阶段。
    pub fn stop(&self, name: &str) {
        if !self.enabled {
            return;
        }
        let Ok(mut stages) = self.stages.lock() else {
            return;
        };
        if let Some(stage) = stages.get_mut(name) {
            stage.elapsed += stage.start.elapsed();
        }
    }

    pub fn elapsed_ms(&self, name: &str) -> i64 {
        let Ok(stages) = self.stages.lock() else {
            return 0;
        };
        stages
            .get(name)
            .map(|stage| stage.elapsed.as_millis() as i64)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_spans_accumulate() {
        let timer = PerfTimer::new(true);
        for _ in 0..3 {
            timer.start("loop");
            std::thread::sleep(Duration::from_millis(5));
            timer.stop("loop");
        }
        // 累加语义要求拿到三段之和；只记最后一段的话约为 5ms。
        assert!(timer.elapsed_ms("loop") >= 12, "阶段耗时应累加");
    }

    #[test]
    fn disabled_timer_reports_zero() {
        let timer = PerfTimer::new(false);
        timer.start("x");
        std::thread::sleep(Duration::from_millis(2));
        timer.stop("x");
        assert_eq!(timer.elapsed_ms("x"), 0);
    }

    #[test]
    fn unknown_stage_reports_zero() {
        assert_eq!(PerfTimer::new(true).elapsed_ms("never-started"), 0);
    }
}
