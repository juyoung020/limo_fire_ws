//! progress.jsonl 키 이름 — 표준 RL 기록 이름(Stable-Baselines3 logger 꼴 `<묶음>/<이름>`)에 탐사 지표를 더한 것.
//! 쓰는 쪽(runfolder.h `rf::key`)과 같은 표다. `runcheck` 가 이 표에 없는 키를 알린다(틀린 철자 잡기).
//!
//! 출처: SB3 logger(time/iterations, time/total_timesteps, time/time_elapsed, time/fps, rollout/ep_rew_mean, rollout/ep_len_mean,
//! rollout/success_rate, train/approx_kl, train/clip_fraction, train/entropy_loss, train/explained_variance, train/learning_rate,
//! train/policy_gradient_loss, train/value_loss, train/std, train/loss, eval/mean_reward, eval/mean_ep_length, eval/success_rate).
//! SB3 에 없는 것은 같은 꼴로 지었다.

/// 정확히 이 이름인 키(이름, 뜻)
pub const KEYS: &[(&str, &str)] = &[
    // time/ — 필수: iterations, total_timesteps(단조 증가)
    ("time/iterations", "PPO 업데이트 번호(필수)"),
    ("time/total_timesteps", "누적 환경 걸음 = 환경 수 × 걸음(필수, 단조 증가)"),
    ("time/time_elapsed", "학습 시작부터 초(재개해도 이어 셈)"),
    ("time/fps", "초당 환경 걸음"),
    ("time/iter_ms", "업데이트 하나 걸린 시간(ms)"),
    ("time/rollout_ms", "롤아웃 시간(ms)"),
    ("time/update_ms", "PPO 갱신 시간(ms)"),
    ("time/iterations_merged", "한 줄에 합친 업데이트 수(runfolder.h 가 넣음)"),
    // rollout/ — 그 사이 끝난 모든 판(모집단). n_episodes 가 0 이면 비율 키는 뺀다
    ("rollout/n_episodes", "이 줄 동안 끝난 판 수(필수 — 가중 평균 분모)"),
    ("rollout/ep_rew_mean", "판 보상 합(리턴) 평균"),
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
    ("rollout/known_area_mean", "판 끝 지도에서 안 넓이(m², 정책이 아는 값) 평균"),
    ("rollout/step_reward_mean", "걸음당 보상 평균"),
    // train/ — PPO
    ("train/policy_gradient_loss", "정책 손실"),
    ("train/value_loss", "가치 손실"),
    ("train/entropy_loss", "엔트로피 손실(= −엔트로피, SB3)"),
    ("train/approx_kl", "근사 KL"),
    ("train/clip_fraction", "잘린 비율"),
    ("train/explained_variance", "설명된 분산"),
    ("train/grad_norm", "기울기 노름(자르기 전)"),
    ("train/learning_rate", "학습률"),
    ("train/loss", "전체 손실"),
    ("train/std", "가우시안 σ 평균(v, vy, ω)"),
    ("train/log_std", "가우시안 log σ 평균"),
    ("train/advantage_std", "이득 표준편차"),
    ("train/advantage_mean", "이득 평균"),
    ("train/value_mean", "가치 평균"),
    ("train/declare_logit_mean", "끝 선언 로짓 평균(시작 −6~−8)"),
    ("train/declare_prob_mean", "걸음당 끝 선언 확률 평균"),
    // eval/ — 평가 판(따로 돈 판)
    ("eval/n_episodes", "평가 판 수"),
    ("eval/mean_reward", "평가 리턴 평균"),
    ("eval/mean_ep_length", "평가 판 길이 평균"),
    ("eval/success_rate", "평가 성공률"),
    ("eval/collision_rate", "평가 충돌률"),
    ("eval/truncation_rate", "평가 끊김 비율"),
    ("eval/declare_short_rate", "평가 기준 미달 선언 비율"),
    ("eval/coverage_mean", "평가 덮은 비율 평균"),
    ("eval/coverage_at_declare_mean", "평가 선언 때 덮은 비율"),
    ("eval/t95_mean", "평가 95 % 도달 시간(s)"),
    ("eval/t95_reached_rate", "평가 95 % 도달 비율"),
    ("eval/path_len_mean", "평가 이동 거리(m)"),
    ("eval/frontier_t95_mean", "같은 맵 프론티어 기준선 95 % 도달 시간(s)"),
    // curriculum/
    ("curriculum/stage", "커리큘럼 단계 번호(0 부터, 이름은 run.json curriculum.stages)"),
    // gpu/
    ("gpu/mem_used_mb", "GPU 메모리 사용(MB)"),
    // ts 는 runfolder.h 가 넣는 유닉스 시각
    ("ts", "줄을 쓴 유닉스 시각"),
];

/// 이 앞부분으로 시작하면 받는 키 묶음(앞부분, 뜻)
pub const FAMILIES: &[(&str, &str)] = &[
    ("reward/", "reward/<항> = 판당 그 보상 항 합 평균. 항 이름은 mdp/rewards.h 가 정한다(run.json reward_terms)"),
    ("eval_reward/", "eval_reward/<항> = 평가 판의 항별 합 평균"),
    ("train/std/", "train/std/<v|vy|w> 행동 성분별 σ"),
    ("train/entropy/", "train/entropy/<v|vy|w|declare> 성분별 엔트로피"),
    ("curriculum/", "curriculum/success_rate/<단계> 등 단계별 값"),
    ("log/", "log/<무엇>_dropped — 버린 기록 수(0 이어야 함)"),
    ("time/", "time/<무엇>_ms 등 시간"),
    ("gpu/", "gpu/<무엇>"),
];

/// 키가 표에 있나(정확히 또는 묶음)
pub fn known(k: &str) -> bool {
    KEYS.iter().any(|(n, _)| *n == k) || FAMILIES.iter().any(|(p, _)| k.starts_with(p) && k.len() > p.len())
}

/// 반드시 있어야 하는 키
pub const REQUIRED: &[&str] = &["time/iterations", "time/total_timesteps"];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn families() {
        assert!(known("reward/coverage"));
        assert!(known("rollout/t95_mean"));
        assert!(!known("reward/"));
        assert!(!known("rollout/succes_rate"));
    }
}
