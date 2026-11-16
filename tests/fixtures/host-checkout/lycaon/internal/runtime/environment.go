package runtime

const (
	EnvConfigRoot        = "LYCAON_CONFIG_ROOT"
	EnvEngineRoot        = "LYCAON_ENGINE_ROOT"
	EnvDecideBinary      = "LYCAON_DECIDE_BINARY"
	EnvDecideDevice      = "LYCAON_DECIDE_DEVICE"
	EnvDecideDisabled    = "LYCAON_DECIDE_DISABLED"
	EnvDecideHeads       = "LYCAON_DECIDE_HEADS"
	EnvDecideModelDir    = "LYCAON_DECIDE_MODEL_DIR"
	EnvMock              = "LYCAON_LLM_MOCK"
	EnvOpengrepCandidate = "LYCAON_OPENGREP_CANDIDATE"
	EnvCaptureDir        = "PW_TRAINING_CAPTURE_DIR"
	EnvCaptureEpoch      = "PW_TRAINING_EPOCH"
	EnvCaptureStudents   = "PW_TRAINING_STUDENTS"

	ContractSchema = "pw-training-contract/1"
	BundleSchema   = "pw-training-episode-bundle/1"
	TurnSchema     = "pw-training-turn/1"
	FactSchema     = "pw-training-fact/1"
	HealthSchema   = "pw-training-health/1"
)
