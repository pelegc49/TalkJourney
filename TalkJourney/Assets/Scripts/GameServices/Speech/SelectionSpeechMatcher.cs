using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Interaction;

namespace TalkJourney.BubbleSystem.Speech
{
    [DisallowMultipleComponent]
    public class SelectionSpeechMatcher : MonoBehaviour
    {
        [Tooltip("Component implementing ISpeechRecognitionService.")]
        public MonoBehaviour speechRecognitionBehaviour;

        [Tooltip("Current active selection bubbles for the stage.")]
        public List<SelectionBubbleController> activeSelectionBubbles = new List<SelectionBubbleController>();

        [Header("Matching")]
        [Tooltip("Try exact bubble match first before using fuzzy distance.")]
        public bool tryExactMatchFirst = true;

        [Tooltip("When exact match fails, use Levenshtein distance against active selection display text.")]
        public bool enableFuzzyMatching = true;

        [Range(0.0f, 1.0f)]
        [Tooltip("Maximum normalized edit distance allowed for fuzzy match. Lower is stricter.")]
        public float maxNormalizedDistance = 0.35f;

        [Min(0)]
        [Tooltip("Minimum allowed absolute edit distance threshold.")]
        public int minimumDistanceThreshold = 1;

        [Header("Bypass")]
        [Tooltip("Button shown after too many failed speech attempts. Clicking it enables selection by click for the current stage.")]
        public VrPointerInteractable bypassButtonInteractable;

        [Min(1)]
        [Tooltip("Failed speech attempts required before bypass button is shown.")]
        public int failedAttemptsBeforeBypass = 3;

        [Tooltip("Hide bypass button until it is unlocked by failed attempts.")]
        public bool hideBypassButtonUntilUnlocked = true;

        [Tooltip("When true, bypass state and failure count reset on each stage change.")]
        public bool resetBypassOnStageChange = true;

        private ISpeechRecognitionService _speechRecognitionService;
        private int[] _levenshteinPrev;
        private int[] _levenshteinCurr;
        private readonly StringBuilder _normalizeBuffer = new StringBuilder(128);
        private int _failedAttempts;
        private bool _isBypassUnlocked;
        private bool _isBypassEnabled;

        private void Awake()
        {
            RefreshDependencies();
            ResetBypassState();
        }

        private void OnEnable()
        {
            if (_speechRecognitionService != null)
            {
                _speechRecognitionService.PhraseRecognized += OnPhraseRecognized;
            }

            if (bypassButtonInteractable != null)
            {
                bypassButtonInteractable.Clicked += OnBypassButtonClicked;
            }

            BubbleEventBus.StageChanged += OnStageChanged;
        }

        private void OnDisable()
        {
            if (_speechRecognitionService != null)
            {
                _speechRecognitionService.PhraseRecognized -= OnPhraseRecognized;
            }

            if (bypassButtonInteractable != null)
            {
                bypassButtonInteractable.Clicked -= OnBypassButtonClicked;
            }

            BubbleEventBus.StageChanged -= OnStageChanged;
        }

        public void SetActiveSelectionBubbles(List<SelectionBubbleController> selectionBubbles)
        {
            activeSelectionBubbles = selectionBubbles ?? new List<SelectionBubbleController>();
            ApplyBypassStateToActiveSelections();
        }

        public void RefreshDependencies()
        {
            _speechRecognitionService = speechRecognitionBehaviour as ISpeechRecognitionService;
            if (_speechRecognitionService == null)
            {
                Debug.LogError("SelectionSpeechMatcher requires speechRecognitionBehaviour implementing ISpeechRecognitionService.", this);
            }
        }

        private void OnPhraseRecognized(string recognizedText)
        {
            BubbleEventBus.PublishSpeechPhraseRecognized(recognizedText);

            if (string.IsNullOrWhiteSpace(recognizedText) || activeSelectionBubbles.Count == 0)
            {
                return;
            }

            var isMatch = false;

            if (tryExactMatchFirst && TryExactMatch(recognizedText))
            {
                isMatch = true;
            }
            else if (enableFuzzyMatching)
            {
                isMatch = TryFuzzyMatch(recognizedText);
            }

            if (!isMatch)
            {
                RegisterFailedSpeechAttempt();
            }
        }

        private void RegisterFailedSpeechAttempt()
        {
            if (_isBypassEnabled)
            {
                return;
            }

            _failedAttempts++;
            if (!_isBypassUnlocked && _failedAttempts >= failedAttemptsBeforeBypass)
            {
                _isBypassUnlocked = true;
                SetBypassButtonVisible(true);
            }
        }

        private void OnBypassButtonClicked()
        {
            if (!_isBypassUnlocked)
            {
                return;
            }

            _isBypassEnabled = true;
            ApplyBypassStateToActiveSelections();
            StartCoroutine(HideBypassButtonAfterClick());
            BubbleEventBus.PublishBypassEnabled();
        }

        private IEnumerator HideBypassButtonAfterClick()
        {
            var animator = bypassButtonInteractable != null ? bypassButtonInteractable.GetComponent<Animator>() : null;
            if (animator != null)
            {
                animator.SetBool("IsClicked", true);
                yield return null;

                var stateInfo = animator.GetCurrentAnimatorStateInfo(0);
                var waitTime = stateInfo.length / Mathf.Max(0.01f, stateInfo.speed);
                if (waitTime > 0f)
                {
                    yield return new WaitForSeconds(waitTime);
                }
            }
            else
            {
                yield return null;
            }

            SetBypassButtonVisible(false);
        }

        private void OnStageChanged(TalkJourney.BubbleSystem.Data.StageData _)
        {
            if (!resetBypassOnStageChange)
            {
                return;
            }

            ResetBypassState();
            ApplyBypassStateToActiveSelections();
        }

        private void ResetBypassState()
        {
            _failedAttempts = 0;
            _isBypassUnlocked = false;
            _isBypassEnabled = false;
            SetBypassButtonVisible(!hideBypassButtonUntilUnlocked);
        }

        private void ApplyBypassStateToActiveSelections()
        {
            for (int i = 0; i < activeSelectionBubbles.Count; i++)
            {
                var bubble = activeSelectionBubbles[i];
                if (bubble != null)
                {
                    bubble.SetBypassEnabled(_isBypassEnabled);
                }
            }
        }

        private void SetBypassButtonVisible(bool isVisible)
        {
            if (bypassButtonInteractable != null)
            {
                bypassButtonInteractable.gameObject.SetActive(isVisible);
            }
        }

        private bool TryExactMatch(string recognizedText)
        {
            for (int i = 0; i < activeSelectionBubbles.Count; i++)
            {
                var selectionBubble = activeSelectionBubbles[i];
                if (selectionBubble == null)
                {
                    continue;
                }

                if (selectionBubble.TryActivateFromRecognizedText(recognizedText))
                {
                    return true;
                }
            }

            return false;
        }

        private bool TryFuzzyMatch(string recognizedText)
        {
            var normalizedRecognized = Normalize(recognizedText);
            if (string.IsNullOrWhiteSpace(normalizedRecognized))
            {
                return false;
            }

            SelectionBubbleController bestMatch = null;
            string bestCandidate = string.Empty;
            int bestDistance = int.MaxValue;

            for (int i = 0; i < activeSelectionBubbles.Count; i++)
            {
                var selectionBubble = activeSelectionBubbles[i];
                if (selectionBubble == null)
                {
                    continue;
                }

                var displayCandidate = Normalize(selectionBubble.GetPrimaryDisplayText());
                var keyCandidate = Normalize(selectionBubble.GetPrimaryKeyText());

                EvaluateCandidate(selectionBubble, normalizedRecognized, displayCandidate, ref bestMatch, ref bestCandidate, ref bestDistance);
                EvaluateCandidate(selectionBubble, normalizedRecognized, keyCandidate, ref bestMatch, ref bestCandidate, ref bestDistance);
            }

            if (bestMatch == null || string.IsNullOrWhiteSpace(bestCandidate))
            {
                return false;
            }

            var threshold = Mathf.Max(minimumDistanceThreshold, Mathf.CeilToInt(bestCandidate.Length * maxNormalizedDistance));
            if (bestDistance > threshold)
            {
                return false;
            }

            return bestMatch.TryActivateFromRecognizedText(bestCandidate);
        }

        private void EvaluateCandidate(
            SelectionBubbleController bubble,
            string normalizedRecognized,
            string normalizedCandidate,
            ref SelectionBubbleController bestMatch,
            ref string bestCandidate,
            ref int bestDistance)
        {
            if (string.IsNullOrWhiteSpace(normalizedCandidate))
            {
                return;
            }

            var distance = LevenshteinDistance(normalizedRecognized, normalizedCandidate);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestMatch = bubble;
                bestCandidate = normalizedCandidate;
            }
        }

        private string Normalize(string value)
        {
            _normalizeBuffer.Clear();

            if (string.IsNullOrWhiteSpace(value))
            {
                return string.Empty;
            }

            var lowered = value.Trim().ToLowerInvariant();
            for (int i = 0; i < lowered.Length; i++)
            {
                var character = lowered[i];
                if (char.IsLetterOrDigit(character) || char.IsWhiteSpace(character))
                {
                    _normalizeBuffer.Append(character);
                }
            }

            return _normalizeBuffer.ToString().Trim();
        }

        private int LevenshteinDistance(string source, string target)
        {
            var sourceLength = source.Length;
            var targetLength = target.Length;

            if (sourceLength == 0)
            {
                return targetLength;
            }

            if (targetLength == 0)
            {
                return sourceLength;
            }

            EnsureLevenshteinCapacity(targetLength + 1);

            for (int j = 0; j <= targetLength; j++)
            {
                _levenshteinPrev[j] = j;
            }

            for (int i = 1; i <= sourceLength; i++)
            {
                _levenshteinCurr[0] = i;
                var sourceChar = source[i - 1];

                for (int j = 1; j <= targetLength; j++)
                {
                    var cost = sourceChar == target[j - 1] ? 0 : 1;
                    var deletion = _levenshteinPrev[j] + 1;
                    var insertion = _levenshteinCurr[j - 1] + 1;
                    var substitution = _levenshteinPrev[j - 1] + cost;
                    _levenshteinCurr[j] = Mathf.Min(Mathf.Min(deletion, insertion), substitution);
                }

                var temp = _levenshteinPrev;
                _levenshteinPrev = _levenshteinCurr;
                _levenshteinCurr = temp;
            }

            return _levenshteinPrev[targetLength];
        }

        private void EnsureLevenshteinCapacity(int requiredLength)
        {
            if (_levenshteinPrev == null || _levenshteinPrev.Length < requiredLength)
            {
                _levenshteinPrev = new int[requiredLength];
                _levenshteinCurr = new int[requiredLength];
            }
        }
    }
}
