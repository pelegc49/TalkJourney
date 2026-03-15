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

        private bool _hasActivated;
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

            if (interactable != null)
            {
                interactable.Clicked += ActivateFromClick;
            }
        }

        private void OnDisable()
        {
            if (interactable != null)
            {
                interactable.Clicked -= ActivateFromClick;
            }
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
            _localizationService = localizationServiceBehaviour as ILocalizationService;
            _stageController = stageControllerBehaviour as IStageController;

            if (interactable == null)
            {
                interactable = GetComponent<VrPointerInteractable>();
            }

            if (_stageController == null)
            {
                Debug.LogError("SelectionBubbleController requires stageControllerBehaviour implementing IStageController.", this);
            }
        }

        private void ActivateFromClick()
        {
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
