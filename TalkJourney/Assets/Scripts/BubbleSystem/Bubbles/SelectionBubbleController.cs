using System.Text;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
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

        [Header("Visuals")]
        [Tooltip("Primary text label shown on the selection bubble.")]
        public TMP_Text primaryText;

        [Tooltip("Optional explicit RectTransform for the primary text. If empty, the TMP text RectTransform is used.")]
        public RectTransform primaryTextRectTransform;

        [Tooltip("Minimum bubble size in pixels.")]
        public Vector2 minimumBubbleSize = new Vector2(50f, 50f);

        [Tooltip("Extra width/height padding added around the primary text in pixels.")]
        public Vector2 bubblePadding = new Vector2(32f, 16f);

        [Tooltip("Extra width/height padding added to the primary text RectTransform around its content.")]
        public Vector2 primaryTextPadding = Vector2.zero;

        [Tooltip("When enabled, the primary text will stop growing wider than its parent and wrap onto multiple lines instead.")]
        public bool constrainPrimaryTextWidthToParent = true;

        [Tooltip("Optional explicit maximum width for the primary text in pixels. Set to 0 to use the parent width.")]
        public float primaryTextMaximumWidth = 0f;

        [Tooltip("When enabled, the selection bubble root is resized to the exact preferred text size, subject to the width cap.")]
        public bool matchPrimaryTextAndBubbleToContent = true;

        [Tooltip("Optional explicit RectTransform for the bubble root. If empty, this component's RectTransform is used.")]
        public RectTransform bubbleRectTransform;

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

        [Header("Selection Feedback")]
        [Tooltip("Animator used to play correct/incorrect feedback animations on this bubble.")]
        public Animator feedbackAnimator;

        [Tooltip("Trigger name for the correct selection feedback animation.")]
        public string correctSelectionTrigger = "Correct";

        [Tooltip("Trigger name for the incorrect selection feedback animation.")]
        public string incorrectSelectionTrigger = "Incorrect";

        [Tooltip("Audio source used for playing correct/incorrect feedback sounds.")]
        public AudioSource feedbackAudioSource;

        [Tooltip("Audio clip played when the selection is correct.")]
        public AudioClip correctSelectionClip;

        [Tooltip("Audio clip played when the selection is incorrect.")]
        public AudioClip incorrectSelectionClip;

        [Tooltip("If enabled, automatically resolve an AudioSource on the same GameObject.")]
        public bool autoResolveFeedbackAudioSource = true;

        private bool _hasActivated;
        private bool _isBypassEnabled;
        private ILocalizationService _localizationService;
        private IStageController _stageController;
        private RectTransform _rectTransform;
        private LayoutElement _layoutElement;
        private readonly StringBuilder _normalizeBuffer = new StringBuilder(128);

        private void Awake()
        {
            _rectTransform = GetComponent<RectTransform>();
            _layoutElement = GetComponent<LayoutElement>();
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

            BubbleEventBus.SelectionCorrect += OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect += OnSelectionIncorrect;

            // Subscribe to language change events
            LocalizationResolver.OnLanguageChanged += OnLanguageChanged;

            RefreshSelectionVisuals();
        }

        private void OnDisable()
        {
            if (interactable != null)
            {
                interactable.Clicked -= ActivateFromClick;
            }

            BubbleEventBus.SelectionCorrect -= OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect -= OnSelectionIncorrect;

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= OnLanguageChanged;
        }

        private void OnLanguageChanged()
        {
            // Language changed - reset activation state and update display if needed
            _hasActivated = false;
            RefreshSelectionVisuals();
        }

        public void Initialize(SelectionBubbleData data)
        {
            selectionData = data;
            RefreshDependencies();
            RefreshSelectionVisuals();
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

            if (feedbackAnimator == null)
            {
                feedbackAnimator = GetComponent<Animator>();
            }

            if (feedbackAudioSource == null && autoResolveFeedbackAudioSource)
            {
                feedbackAudioSource = GetComponent<AudioSource>();
            }

            if (primaryText == null)
            {
                primaryText = GetComponentInChildren<TMP_Text>(true);
            }

            if (primaryTextRectTransform == null && primaryText != null)
            {
                primaryTextRectTransform = primaryText.rectTransform;
            }

            if (bubbleRectTransform == null)
            {
                bubbleRectTransform = _rectTransform != null ? _rectTransform : GetComponent<RectTransform>();
            }

            if (_layoutElement == null)
            {
                _layoutElement = GetComponent<LayoutElement>();
            }

            if (_stageController == null)
            {
                _stageController = FindFirstObjectByType<StageController>(FindObjectsInactive.Include);
            }

            if (_stageController == null && selectionData != null)
            {
                Debug.LogWarning("SelectionBubbleController could not resolve IStageController. Selection activation will be disabled.", this);
            }

            RefreshSelectionVisuals();
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
            if (_stageController == null || selectionData == null)
            {
                return false;
            }

            if (preventDuplicateActivation && _hasActivated)
            {
                return false;
            }

            var didHandleSelection = _stageController.TryHandleSelection(selectionData);
            if (didHandleSelection)
            {
                _hasActivated = true;
            }

            return didHandleSelection;
        }

        private void OnSelectionCorrect(SelectionBubbleData selectedData)
        {
            if (selectedData == selectionData)
            {
                PlaySelectionFeedback(true);
            }
        }

        private void OnSelectionIncorrect(SelectionBubbleData selectedData)
        {
            if (selectedData == selectionData)
            {
                PlaySelectionFeedback(false);
            }
        }

        private void PlaySelectionFeedback(bool isCorrect)
        {
            if (feedbackAnimator != null)
            {
                var trigger = isCorrect ? correctSelectionTrigger : incorrectSelectionTrigger;
                if (!string.IsNullOrEmpty(trigger))
                {
                    feedbackAnimator.SetTrigger(trigger);
                }
            }

            if (feedbackAudioSource != null)
            {
                var clip = isCorrect ? correctSelectionClip : incorrectSelectionClip;
                if (clip != null)
                {
                    feedbackAudioSource.PlayOneShot(clip);
                }
            }
        }

        public void RefreshSelectionVisuals()
        {
            if (selectionData == null)
            {
                return;
            }

            if (primaryText == null)
            {
                primaryText = GetComponentInChildren<TMP_Text>(true);
            }

            if (primaryText == null)
            {
                return;
            }

            if (_layoutElement == null)
            {
                _layoutElement = GetComponent<LayoutElement>();
                if (_layoutElement == null)
                {
                    _layoutElement = gameObject.AddComponent<LayoutElement>();
                }
            }

            if (primaryTextRectTransform == null)
            {
                primaryTextRectTransform = primaryText.rectTransform;
            }

            if (bubbleRectTransform == null)
            {
                bubbleRectTransform = _rectTransform != null ? _rectTransform : GetComponent<RectTransform>();
            }

            var resolvedText = ResolvePrimary(selectionData.bubble != null ? selectionData.bubble.primaryTextKey : null);
            primaryText.text = resolvedText;
            primaryText.ForceMeshUpdate();

            var shouldConstrainWidth = constrainPrimaryTextWidthToParent || primaryTextMaximumWidth > 0f;
            primaryText.textWrappingMode = shouldConstrainWidth ? TextWrappingModes.Normal : TextWrappingModes.NoWrap;
            primaryText.overflowMode = TextOverflowModes.Overflow;

            var textWidthLimit = GetPrimaryTextWidthLimit();
            var preferred = shouldConstrainWidth
                ? primaryText.GetPreferredValues(primaryText.text, textWidthLimit, Mathf.Infinity)
                : primaryText.GetPreferredValues(primaryText.text, Mathf.Infinity, Mathf.Infinity);

            var textWidth = Mathf.Max(0f, preferred.x + primaryTextPadding.x);
            var textHeight = Mathf.Max(0f, preferred.y + primaryTextPadding.y);

            if (shouldConstrainWidth && !float.IsInfinity(textWidthLimit))
            {
                textWidth = Mathf.Min(textWidth, textWidthLimit);
            }

            if (primaryTextRectTransform != null)
            {
                primaryTextRectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, textWidth);
                primaryTextRectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, textHeight);
            }

            if (!matchPrimaryTextAndBubbleToContent)
            {
                return;
            }

            var bubbleWidth = Mathf.Max(minimumBubbleSize.x, textWidth + bubblePadding.x);
            var bubbleHeight = Mathf.Max(minimumBubbleSize.y, textHeight + bubblePadding.y);

            if (bubbleRectTransform != null)
            {
                bubbleRectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, bubbleWidth);
                bubbleRectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, bubbleHeight);
            }

            _layoutElement.preferredWidth = bubbleWidth;
            _layoutElement.preferredHeight = bubbleHeight;
            _layoutElement.minWidth = minimumBubbleSize.x;
            _layoutElement.minHeight = minimumBubbleSize.y;

            if (_rectTransform != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(_rectTransform);
            }

            var parentRect = transform.parent as RectTransform;
            if (parentRect != null)
            {
                LayoutRebuilder.ForceRebuildLayoutImmediate(parentRect);
            }
        }

        private float GetPrimaryTextWidthLimit()
        {
            var widthLimit = primaryTextMaximumWidth > 0f ? primaryTextMaximumWidth : Mathf.Infinity;

            if (!constrainPrimaryTextWidthToParent || primaryTextRectTransform == null)
            {
                return widthLimit;
            }

            var parentRect = primaryTextRectTransform.parent as RectTransform;
            if (parentRect == null)
            {
                return widthLimit;
            }

            var parentWidthLimit = Mathf.Max(0f, parentRect.rect.width - bubblePadding.x);
            if (parentWidthLimit <= 0f)
            {
                return widthLimit;
            }

            return float.IsInfinity(widthLimit) ? parentWidthLimit : Mathf.Min(widthLimit, parentWidthLimit);
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
