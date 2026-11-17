//! progress.jsonl 키 이름 — `<묶음>/<이름>` 꼴(Stable-Baselines3 logger 이름 꼴)에 탐사 지표를 더한 것.
//! 쓰는 쪽(runfolder.h `rf::key`)과 같은 표다. `runcheck` 가 이 표에 없는 키를 알린다(틀린 철자 잡기).
//!
//! 2026-11-18: 강화학습을 그만두며(archive/rl) PPO·보상·커리큘럼·자원 키(train/·reward/·eval_reward/·curriculum/·gpu/·log/,
//! time/fps·*_ms, rollout/ep_rew_mean 등)를 뺐다. 남은 것은 프론티어 기준선(frontier_run)이 쓰거나 뷰어가 읽는 키뿐이다.
//! 옛 RL 실행 폴더를 runcheck 하면 그 키들은 경고(모르는 키)로 나온다 — 오류는 아니다.

/// 정확히 이 이름인 키(이름, 뜻)
pub const KEYS: &[(&str, &str)] = &[
    // time/ — 필수: iterations, total_timesteps(단조 증가)
    ("time/iterations", "줄 번호(필수, 단조 증가) — frontier_run 은 장면 번호"),
    ("time/total_timesteps", "누적 환경 걸음(필수, 단조 증가)"),
    ("time/time_elapsed", "시작부터 초(재개해도 이어 셈, 안 넣으면 runfolder.h 가 넣음)"),
    ("time/iterations_merged", "한 줄에 합친 업데이트 수(runfolder.h 가 넣음)"),
    // rollout/ — 그 사이 끝난 모든 판(모집단). n_episodes 가 0 이면 비율 키는 뺀다
    ("rollout/n_episodes", "이 줄 동안 끝난 판 수(가중 평균 분모)"),
    ("rollout/ep_len_mean", "판 길이(걸음) 평균"),
    ("rollout/success_rate", "선언 + 기준 덮음 이상 비율"),
    ("rollout/collision_rate", "충돌로 끝난 비율"),
    ("rollout/truncation_rate", "안전 시간으로 끊긴 비율"),
    ("rollout/declare_rate", "끝 선언으로 끝난 비율(성공 + 미달)"),
    ("rollout/declare_short_rate", "기준 미달 선언 비율"),
    ("rollout/coverage_mean", "판 끝 덮은 비율(정답 맵 기준) 평균"),
    ("rollout/coverage_at_declare_mean", "선언한 판의 선언 때 덮은 비율 평균"),
    ("rollout/t95_mean", "95 % 덮음 도달 시간(s) 평균 — 도달한 판만"),
    ("rollout/t95_reached_rate", "95 % 덮음에 도달한 판 비율"),
    ("rollout/path_len_mean", "이동 거리(m) 평균"),
    ("rollout/known_area_mean", "판 끝 지도에서 안 넓이(m²) 평균"),
    // eval/
    ("eval/frontier_t95_mean", "같은 맵 프론티어 기준선 95 % 도달 시간(s) — frontier_run 이 씀"),
    // ts 는 runfolder.h 가 넣는 유닉스 시각
    ("ts", "줄을 쓴 유닉스 시각"),
];

/// 키가 표에 있나
pub fn known(k: &str) -> bool {
    KEYS.iter().any(|(n, _)| *n == k)
}

/// 반드시 있어야 하는 키
pub const REQUIRED: &[&str] = &["time/iterations", "time/total_timesteps"];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_keys() {
        assert!(known("rollout/t95_mean"));
        assert!(known("eval/frontier_t95_mean"));
        assert!(!known("rollout/succes_rate"));
        assert!(!known("train/approx_kl"));
    }
}
