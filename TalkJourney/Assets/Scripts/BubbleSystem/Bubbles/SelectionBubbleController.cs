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

        [Tooltip("Root object of TransliteratorText_UI shown only while hovering.")]
        public GameObject transliteratorTextObject;

        [Tooltip("Optional animator controlling transliterator show/hide. If assigned, hover enter/exit will fire triggers instead of instant disable.")]
        public Animator transliteratorAnimator;

        [Tooltip("Animator bool parameter name used for transliterator visibility.")]
        public string transliteratorVisibleBool = "IsVisible";

        [Tooltip("Optional explicit RectTransform for the primary text. If empty, the TMP text RectTransform is used.")]
        public RectTransform primaryTextRectTransform;

        [Tooltip("Minimum bubble size in pixels.")]
        public Vector2 minimumBubbleSize = new Vector2(50f, 50f);

        [Tooltip("Extra width/height padding added around the primary text in pixels.")]
        public Vector2 bubblePadding = new Vector2(32f, 16f);

        [Tooltip("When enabled, TransliteratorText_UI RectTransform is resized to the exact preferred text size and the transliterator bubble is matched 1:1 to it.")]
        public bool matchTransliteratorTextAndBubbleToContent = true;

        [Tooltip("Extra width/height padding added to the primary text RectTransform around its content.")]
        public Vector2 primaryTextPadding = Vector2.zero;

        [Tooltip("Optional explicit RectTransform for TransliteratorText_UI. If empty, TMP_Text on transliteratorTextObject is used.")]
        public RectTransform transliteratorTextRectTransform;

        [Tooltip("Extra width/height padding added to TransliteratorText_UI around its content.")]
        public Vector2 transliteratorTextPadding = Vector2.zero;

        [Tooltip("Optional explicit RectTransform for TransliteratorBubble image. If empty, it is auto-resolved under TransliteratorText_UI/TransliteratorBubble.")]
        public RectTransform transliteratorBubbleRectTransform;

        [Tooltip("Extra width/height padding added to TransliteratorBubble around TransliteratorText_UI.")]
        public Vector2 transliteratorBubblePadding = Vector2.zero;

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
        private bool _isTransliteratorVisible;
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
                interactable.HoverEntered += OnHoverEntered;
                interactable.HoverExited += OnHoverExited;
            }

            BubbleEventBus.SelectionCorrect += OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect += OnSelectionIncorrect;

            // Subscribe to language change events
            LocalizationResolver.OnLanguageChanged += OnLanguageChanged;
            LocalizationResolver.OnTransliteratorChanged += RefreshTransliteratorText;

            RefreshSelectionVisuals();
            SetTransliteratorVisible(false, immediate: true);
        }

        private void OnDisable()
        {
            if (interactable != null)
            {
                interactable.Clicked -= ActivateFromClick;
                interactable.HoverEntered -= OnHoverEntered;
                interactable.HoverExited -= OnHoverExited;
            }

            BubbleEventBus.SelectionCorrect -= OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect -= OnSelectionIncorrect;

            // Unsubscribe from language change events
            LocalizationResolver.OnLanguageChanged -= OnLanguageChanged;
            LocalizationResolver.OnTransliteratorChanged -= RefreshTransliteratorText;
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
            SetTransliteratorVisible(false, immediate: true);
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

            if (transliteratorAnimator == null)
            {
                transliteratorAnimator = feedbackAnimator;
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

            RefreshTransliteratorText();

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

        private void OnHoverEntered()
        {
            SetTransliteratorVisible(true);
        }

        private void OnHoverExited()
        {
            SetTransliteratorVisible(false);
        }

        private void RefreshTransliteratorText()
        {
            var transliteratorText = ResolveTransliteratorTextComponent();
            if (selectionData == null || transliteratorText == null)
            {
                return;
            }

            transliteratorText.text = ResolveTransliterator(selectionData.bubble != null ? selectionData.bubble.primaryTextKey : null);
            transliteratorText.ForceMeshUpdate();

            ApplyTransliteratorPreferredBubbleSize();
        }

        private void ApplyTransliteratorPreferredBubbleSize()
        {
            if (!matchTransliteratorTextAndBubbleToContent)
            {
                return;
            }

            ResizeTransliteratorTextAndBubbleToContent();
        }

        private Vector2 ResizeTransliteratorTextAndBubbleToContent()
        {
            var transliteratorText = ResolveTransliteratorTextComponent();
            var transliteratorRect = ResolveTransliteratorTextRectTransform();
            if (transliteratorText == null || transliteratorRect == null)
            {
                return Vector2.zero;
            }

            var preferred = transliteratorText.GetPreferredValues(transliteratorText.text, Mathf.Infinity, Mathf.Infinity);
            var textWidth = Mathf.Max(0f, preferred.x + transliteratorTextPadding.x);
            var textHeight = Mathf.Max(0f, preferred.y + transliteratorTextPadding.y);

            transliteratorRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, textWidth);
            transliteratorRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, textHeight);

            var bubbleWidth = Mathf.Max(0f, textWidth + transliteratorBubblePadding.x);
            var bubbleHeight = Mathf.Max(0f, textHeight + transliteratorBubblePadding.y);

            var transliteratorBubbleRect = ResolveTransliteratorBubbleRectTransform(transliteratorRect);
            if (transliteratorBubbleRect != null)
            {
                transliteratorBubbleRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, bubbleWidth);
                transliteratorBubbleRect.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, bubbleHeight);
            }

            return new Vector2(Mathf.Max(textWidth, bubbleWidth), Mathf.Max(textHeight, bubbleHeight));
        }

        private RectTransform ResolveTransliteratorTextRectTransform()
        {
            if (transliteratorTextRectTransform != null)
            {
                return transliteratorTextRectTransform;
            }

            var transliteratorText = ResolveTransliteratorTextComponent();
            if (transliteratorText != null)
            {
                return transliteratorText.rectTransform;
            }

            return null;
        }

        private TMP_Text ResolveTransliteratorTextComponent()
        {
            if (transliteratorTextObject == null)
            {
                return null;
            }

            return transliteratorTextObject.GetComponent<TMP_Text>();
        }

        private RectTransform ResolveTransliteratorBubbleRectTransform(RectTransform transliteratorRect)
        {
            if (transliteratorBubbleRectTransform != null)
            {
                return transliteratorBubbleRectTransform;
            }

            if (transliteratorRect == null)
            {
                return null;
            }

            var transliteratorBubble = transliteratorRect.Find("TransliteratorBubble") as RectTransform;
            if (transliteratorBubble != null)
            {
                return transliteratorBubble;
            }

            var image = transliteratorRect.GetComponentInChildren<Image>(true);
            return image != null ? image.rectTransform : null;
        }

        private void SetTransliteratorVisible(bool isVisible, bool immediate = false)
        {
            _isTransliteratorVisible = isVisible;

            var hasAnimator = transliteratorAnimator != null && !string.IsNullOrWhiteSpace(transliteratorVisibleBool);
            if (hasAnimator)
            {
                SetTransliteratorObjectsActive(true);
                transliteratorAnimator.SetBool(transliteratorVisibleBool, isVisible);

                if (!isVisible && immediate)
                {
                    SetTransliteratorObjectsActive(false);
                }

                return;
            }

            SetTransliteratorObjectsActive(isVisible);
        }

        private void SetTransliteratorObjectsActive(bool isActive)
        {
            if (transliteratorTextObject != null)
            {
                transliteratorTextObject.SetActive(isActive);
            }
        }

        private string ResolveTransliterator(string key)
        {
            if (_localizationService == null || string.IsNullOrWhiteSpace(key))
            {
                return key ?? string.Empty;
            }

            var localizationResolver = localizationServiceBehaviour as LocalizationResolver;
            if (localizationResolver == null)
            {
                localizationResolver = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
            }

            if (localizationResolver != null)
            {
                var transliteratorCode = localizationResolver.GetCurrentTransliteratorCode();
                if (!string.IsNullOrWhiteSpace(transliteratorCode)
                    && _localizationService.TryResolveForLocaleCode(key, transliteratorCode, out var transliteratedValue))
                {
                    return transliteratedValue;
                }
            }

            return _localizationService.Resolve(key);
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
