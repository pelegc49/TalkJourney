using System.Text;
using UnityEngine;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Flow;
using TalkJourney.BubbleSystem.Interaction;
using TalkJourney.BubbleSystem.Localization;
using TalkJourney.BubbleSystem.Speech;

namespace TalkJourney.BubbleSystem.Bubbles
{
    [DisallowMultipleComponent]
    public class SelectionBubbleController : MonoBehaviour, ISelectionSpeechTarget
    {
        [Header("Data")]
        public SelectionBubbleData selectionData;

        [Header("Dependencies")]
        [Tooltip("Pointer interaction adapter for click handling.")]
        public VrPointerInteractable interactable;

        [Tooltip("Component implementing ILocalizationService.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Component implementing IStageController.")]
        public MonoBehaviour stageControllerBehaviour;

        [Header("Activation")]
        [Tooltip("Prevents duplicate transitions from nearly simultaneous click and speech events.")]
        public bool preventDuplicateActivation = true;

        [Tooltip("When enabled, selection clicks only work after bypass is enabled by the speech matcher.")]
        public bool requireBypassForClick = true;

        private bool _hasActivated;
        private bool _isBypassEnabled;
        private ILocalizationService _localizationService;
        private IStageController _stageController;
        private readonly StringBuilder _normalizeBuffer = new StringBuilder(128);

        private void Awake()
        {
            RefreshDependencies();
        }

        private void OnEnable()
        {
            _hasActivated = false;
            _isBypassEnabled = false;

            if (interactable != null)
            {
                interactable.Clicked += ActivateFromClick;
            }

            // Subscribe to language change events
            LocalizationResolver.OnLanguageChanged += OnLanguageChanged;
        }

        private void OnDisable()
        {
            if (interactable != null)
            {
                interactable.Clicked -= ActivateFromClick;
            }

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= OnLanguageChanged;
        }

        private void OnLanguageChanged()
        {
            // Language changed - reset activation state and update display if needed
            _hasActivated = false;
        }

        public bool TryActivateFromRecognizedText(string recognizedText)
        {
            if (!MatchesRecognizedText(recognizedText))
            {
                return false;
            }

            return ActivateSelection();
        }

        public string GetPrimaryDisplayText()
        {
            if (selectionData == null || selectionData.bubble == null)
            {
                return string.Empty;
            }

            return ResolvePrimary(selectionData.bubble.primaryTextKey);
        }

        public string GetPrimaryKeyText()
        {
            if (selectionData == null || selectionData.bubble == null)
            {
                return string.Empty;
            }

            return selectionData.bubble.primaryTextKey ?? string.Empty;
        }

        public void RefreshDependencies()
        {
            if (stageControllerBehaviour == null)
            {
                stageControllerBehaviour = GetComponentInParent<StageController>(true);
            }

            if (localizationServiceBehaviour == null)
            {
                localizationServiceBehaviour = GetComponentInParent<LocalizationResolver>(true);
            }

            _localizationService = localizationServiceBehaviour as ILocalizationService;
            _stageController = stageControllerBehaviour as IStageController;

            if (interactable == null)
            {
                interactable = GetComponent<VrPointerInteractable>();
            }

            if (_stageController == null)
            {
                _stageController = FindFirstObjectByType<StageController>(FindObjectsInactive.Include);
            }

            if (_stageController == null && selectionData != null)
            {
                Debug.LogWarning("SelectionBubbleController could not resolve IStageController. Selection activation will be disabled.", this);
            }
        }

        public void SetBypassEnabled(bool isEnabled)
        {
            _isBypassEnabled = isEnabled;
        }

        private void ActivateFromClick()
        {
            if (requireBypassForClick && !_isBypassEnabled)
            {
                return;
            }

            ActivateSelection();
        }

        private bool ActivateSelection()
        {
            if (_stageController == null || selectionData == null || selectionData.nextStage == null)
            {
                return false;
            }

            if (preventDuplicateActivation && _hasActivated)
            {
                return false;
            }

            var didTransition = _stageController.TransitionToStage(selectionData.nextStage);
            if (didTransition)
            {
                _hasActivated = true;
            }

            return didTransition;
        }

        private bool MatchesRecognizedText(string recognizedText)
        {
            if (selectionData == null || selectionData.bubble == null)
            {
                return false;
            }

            var normalizedRecognized = Normalize(recognizedText);
            if (string.IsNullOrEmpty(normalizedRecognized))
            {
                return false;
            }

            var primaryKey = selectionData.bubble.primaryTextKey;
            var localizedPrimary = ResolvePrimary(primaryKey);

            return normalizedRecognized == Normalize(localizedPrimary)
                || normalizedRecognized == Normalize(primaryKey);
        }

        private string ResolvePrimary(string key)
        {
            if (_localizationService == null || string.IsNullOrWhiteSpace(key))
            {
                return key ?? string.Empty;
            }

            return _localizationService.Resolve(key);
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
    }
}
