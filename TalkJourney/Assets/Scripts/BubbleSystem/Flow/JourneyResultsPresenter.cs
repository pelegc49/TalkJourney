using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Bubbles;
using TalkJourney.BubbleSystem.Events;
using TalkJourney.BubbleSystem.Localization;
using TMPro;
using UnityEngine;
using System.Collections.Generic;

namespace TalkJourney.BubbleSystem.Flow
{
    /// <summary>
    /// Listens to BubbleEventBus and presents a simple journey results screen.
    /// Attach to a persistent scene object and assign results panel/text references.
    /// </summary>
    [DisallowMultipleComponent]
    public class JourneyResultsPresenter : MonoBehaviour
    {
        [Header("UI")]
        [Tooltip("Panel/canvas root shown when journey is completed.")]
        public GameObject resultsPanelRoot;

        [Tooltip("Optional close button root that follows the same visibility as resultsPanelRoot.")]
        public GameObject closeButtonRoot;

        [Tooltip("If enabled, render the results as display bubbles using localization/transliterator flow.")]
        public bool useBubbleResults = true;

        [Tooltip("Parent where result bubbles are instantiated.")]
        public Transform resultsBubbleParent;

        [Tooltip("Optional row prefab for label/value bubble pairs. If assigned, metrics are rendered by rows.")]
        public ResultRowView resultRowPrefab;

        [Tooltip("Bubble prefab used for result lines.")]
        public DisplayBubbleController resultBubblePrefab;

        [Tooltip("Localization resolver used by spawned result bubbles.")]
        public MonoBehaviour localizationServiceBehaviour;

        [Tooltip("Optional audio manager used by spawned result bubbles.")]
        public MonoBehaviour audioPlaybackManagerBehaviour;

        [Tooltip("Optional title/status text.")]
        public TMP_Text completionStatusText;

        [Header("Animation")]
        [Tooltip("Optional animator used to play the results fade-in animation.")]
        public Animator resultsAnimator;

        [Tooltip("Animator bool parameter used to show results.")]
        public string resultsShowBool = "IsVisible";

        [Tooltip("Animator trigger parameter used to show results.")]
        public string resultsShowTrigger = "FadeIn";

        public TMP_Text durationText;
        public TMP_Text totalSelectionsText;
        public TMP_Text correctSelectionsText;
        public TMP_Text incorrectSelectionsText;
        public TMP_Text accuracyText;

        [Header("Formatting")]
        [Tooltip("Duration format in mm:ss.")]
        public string durationPrefix = "Duration: ";

        [Tooltip("Localization key used for completion status bubble.")]
        public string completionStatusKey = "results.level_complete";

        [Tooltip("Localization key used for duration label bubble.")]
        public string durationLabelKey = "results.duration";

        [Tooltip("Localization key used for total selections label bubble.")]
        public string totalSelectionsLabelKey = "results.total_selections";

        [Tooltip("Localization key used for correct selections label bubble.")]
        public string correctSelectionsLabelKey = "results.correct";

        [Tooltip("Localization key used for incorrect selections label bubble.")]
        public string incorrectSelectionsLabelKey = "results.incorrect";

        [Tooltip("Localization key used for accuracy label bubble.")]
        public string accuracyLabelKey = "results.accuracy";

        [Tooltip("If true, panel is hidden on Awake.")]
        public bool hidePanelOnAwake = true;

        [Header("Close Button Behavior")]
        [Tooltip("If enabled, closing results also requests BubbleSystemLauncher to destroy runtime-created containers.")]
        public bool destroyRuntimeContainersOnClose = true;

        [Tooltip("Optional launcher used to destroy runtime-created bubble/results containers on close.")]
        public BubbleSystemLauncher bubbleSystemLauncher;

        [Tooltip("If enabled, auto-resolves BubbleSystemLauncher when reference is empty.")]
        public bool autoResolveBubbleSystemLauncher = true;

        private readonly JourneySessionStats _stats = new JourneySessionStats();
        private readonly List<DisplayBubbleController> _spawnedResultBubbles = new List<DisplayBubbleController>();
        private readonly List<BubbleData> _runtimeBubbleData = new List<BubbleData>();
        private readonly List<ResultRowView> _spawnedRows = new List<ResultRowView>();
        private ILocalizationService _localizationService;
        private bool _pendingShowResults;

        private void Awake()
        {
            _stats.BeginSession();
            RefreshDependencies();
            ResolveBubbleSystemLauncher();

            if (hidePanelOnAwake)
            {
                SetResultsVisibility(false);
            }
        }

        private void OnEnable()
        {
            BubbleEventBus.StageChanged += OnStageChanged;
            BubbleEventBus.SelectionCorrect += OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect += OnSelectionIncorrect;
            BubbleEventBus.JourneyCompleted += OnJourneyCompleted;
            BubbleEventBus.BubbleSystemHidden += OnBubbleSystemHidden;
            LocalizationResolver.OnLanguageChanged += OnLanguageChanged;
            LocalizationResolver.OnTransliteratorChanged += OnTransliteratorChanged;
        }

        private void OnDisable()
        {
            BubbleEventBus.StageChanged -= OnStageChanged;
            BubbleEventBus.SelectionCorrect -= OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect -= OnSelectionIncorrect;
            BubbleEventBus.JourneyCompleted -= OnJourneyCompleted;
            BubbleEventBus.BubbleSystemHidden -= OnBubbleSystemHidden;
            LocalizationResolver.OnLanguageChanged -= OnLanguageChanged;
            LocalizationResolver.OnTransliteratorChanged -= OnTransliteratorChanged;
        }

        public void HideResultsAndStartNewSession()
        {
            SetResultsVisibility(false);

            ClearResultBubbles();

            _stats.BeginSession();
        }

        /// <summary>
        /// Intended for Close button OnClick: hides results and optionally destroys runtime containers via launcher.
        /// </summary>
        public void OnCloseButtonClicked()
        {
            HideResultsAndStartNewSession();

            if (!destroyRuntimeContainersOnClose)
            {
                return;
            }

            var launcher = ResolveBubbleSystemLauncher();
            if (launcher == null)
            {
                return;
            }

            launcher.StopAndDestroyActiveBootstrap();
        }

        private void OnStageChanged(StageData stageData)
        {
            _stats.RegisterStage(stageData);
        }

        private void OnSelectionCorrect(SelectionBubbleData _)
        {
            _stats.RegisterCorrectSelection();
        }

        private void OnSelectionIncorrect(SelectionBubbleData _)
        {
            _stats.RegisterIncorrectSelection();
        }

        private void OnJourneyCompleted(StageData _, SelectionBubbleData __)
        {
            _stats.Complete();
            RefreshUI();
            _pendingShowResults = true;
        }

        private void OnBubbleSystemHidden()
        {
            if (!_pendingShowResults)
            {
                return;
            }

            SetResultsVisibility(true);
            _pendingShowResults = false;
        }

        private void RefreshUI()
        {
            RebuildResultBubbles();

            if (completionStatusText != null)
            {
                completionStatusText.text = ResolveOrFallback(completionStatusKey, "Level Complete");
            }

            if (durationText != null)
            {
                var duration = _stats.Duration;
                var durationLabel = ResolveOrFallback(durationLabelKey, "Duration");
                durationText.text = durationLabel + ": " + duration.ToString(@"mm\:ss");
            }

            if (totalSelectionsText != null)
            {
                var totalLabel = ResolveOrFallback(totalSelectionsLabelKey, "Total Selections");
                totalSelectionsText.text = totalLabel + ": " + _stats.TotalSelections;
            }

            if (correctSelectionsText != null)
            {
                var correctLabel = ResolveOrFallback(correctSelectionsLabelKey, "Correct");
                correctSelectionsText.text = correctLabel + ": " + _stats.CorrectSelections;
            }

            if (incorrectSelectionsText != null)
            {
                var incorrectLabel = ResolveOrFallback(incorrectSelectionsLabelKey, "Incorrect");
                incorrectSelectionsText.text = incorrectLabel + ": " + _stats.IncorrectSelections;
            }

            if (accuracyText != null)
            {
                var accuracyLabel = ResolveOrFallback(accuracyLabelKey, "Accuracy");
                accuracyText.text = accuracyLabel + ": " + (_stats.Accuracy * 100f).ToString("0") + "%";
            }


        }

        private void RefreshDependencies()
        {
            if (localizationServiceBehaviour == null)
            {
                var globalServices = GameServices.GlobalGameServicesBootstrap.Instance;
                if (globalServices != null && globalServices.localizationResolver != null)
                {
                    localizationServiceBehaviour = globalServices.localizationResolver;
                }
                else
                {
                    localizationServiceBehaviour = FindFirstObjectByType<LocalizationResolver>(FindObjectsInactive.Include);
                }
            }

            _localizationService = localizationServiceBehaviour as ILocalizationService;
        }

        private void OnLanguageChanged()
        {
            if (resultsPanelRoot != null && resultsPanelRoot.activeInHierarchy)
            {
                RefreshUI();
            }
        }

        private void OnTransliteratorChanged()
        {
            if (resultsPanelRoot != null && resultsPanelRoot.activeInHierarchy)
            {
                RefreshUI();
            }
        }

        private void RebuildResultBubbles()
        {
            if (!useBubbleResults || resultsBubbleParent == null || resultBubblePrefab == null)
            {
                ClearResultBubbles();
                return;
            }

            ClearResultBubbles();

            if (resultRowPrefab != null)
            {
                AddSingleBubbleRow(completionStatusKey);
                AddMetricRow(durationLabelKey, _stats.Duration.ToString(@"mm\:ss"));
                AddMetricRow(totalSelectionsLabelKey, _stats.TotalSelections.ToString());
                AddMetricRow(correctSelectionsLabelKey, _stats.CorrectSelections.ToString());
                AddMetricRow(incorrectSelectionsLabelKey, _stats.IncorrectSelections.ToString());
                AddMetricRow(accuracyLabelKey, (_stats.Accuracy * 100f).ToString("0") + "%");
                return;
            }

            AddLocalizedResultBubble(completionStatusKey);
            AddLocalizedResultBubble(durationLabelKey);
            AddRawResultBubble(_stats.Duration.ToString(@"mm\:ss"));
            AddLocalizedResultBubble(totalSelectionsLabelKey);
            AddRawResultBubble(_stats.TotalSelections.ToString());
            AddLocalizedResultBubble(correctSelectionsLabelKey);
            AddRawResultBubble(_stats.CorrectSelections.ToString());
            AddLocalizedResultBubble(incorrectSelectionsLabelKey);
            AddRawResultBubble(_stats.IncorrectSelections.ToString());
            AddLocalizedResultBubble(accuracyLabelKey);
            AddRawResultBubble((_stats.Accuracy * 100f).ToString("0") + "%");
        }

        private void AddSingleBubbleRow(string labelKey)
        {
            var row = SpawnRow();
            if (row == null)
            {
                AddLocalizedResultBubble(labelKey);
                return;
            }

            SpawnLocalizedBubbleInto(row.ResolveLabelParent(), labelKey);
        }

        private void AddMetricRow(string labelKey, string valueText)
        {
            var row = SpawnRow();
            if (row == null)
            {
                AddLocalizedResultBubble(labelKey);
                AddRawResultBubble(valueText);
                return;
            }

            if (row.HasSeparateSlots)
            {
                SpawnLocalizedBubbleInto(row.ResolveLabelParent(), labelKey);
                SpawnRawBubbleInto(row.ResolveValueParent(), valueText);
                return;
            }

            // When no explicit slots are assigned, both bubbles are added to row root and laid out by row layout group.
            SpawnLocalizedBubbleInto(row.transform, labelKey);
            SpawnRawBubbleInto(row.transform, valueText);
        }

        private ResultRowView SpawnRow()
        {
            if (resultRowPrefab == null || resultsBubbleParent == null)
            {
                return null;
            }

            var row = Instantiate(resultRowPrefab, resultsBubbleParent);
            _spawnedRows.Add(row);
            return row;
        }

        private void AddLocalizedResultBubble(string key)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                return;
            }

            var data = new BubbleData
            {
                primaryTextKey = key,
                visualElementPrefab = resultBubblePrefab.gameObject
            };

            SpawnResultBubble(data);
        }

        private void AddRawResultBubble(string rawText)
        {
            if (string.IsNullOrWhiteSpace(rawText))
            {
                return;
            }

            var data = new BubbleData
            {
                // Raw content bubble for dynamic values like numbers and path.
                primaryTextKey = rawText.Trim(),
                visualElementPrefab = resultBubblePrefab.gameObject
            };

            SpawnResultBubble(data);
        }

        private void SpawnResultBubble(BubbleData data)
        {
            SpawnResultBubble(resultsBubbleParent, data);
        }

        private void SpawnLocalizedBubbleInto(Transform parent, string key)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                return;
            }

            var data = new BubbleData
            {
                primaryTextKey = key,
                visualElementPrefab = resultBubblePrefab.gameObject
            };

            SpawnResultBubble(parent, data);
        }

        private void SpawnRawBubbleInto(Transform parent, string rawText)
        {
            if (string.IsNullOrWhiteSpace(rawText))
            {
                return;
            }

            var data = new BubbleData
            {
                primaryTextKey = rawText.Trim(),
                visualElementPrefab = resultBubblePrefab.gameObject
            };

            SpawnResultBubble(parent, data);
        }

        private void SpawnResultBubble(Transform parent, BubbleData data)
        {
            if (data == null)
            {
                return;
            }

            if (parent == null)
            {
                return;
            }

            var instance = Instantiate(resultBubblePrefab, parent);
            instance.transform.localScale = Vector3.one;
            instance.localizationServiceBehaviour = localizationServiceBehaviour;
            instance.audioPlaybackManagerBehaviour = audioPlaybackManagerBehaviour;
            instance.RefreshDependencies();
            instance.Initialize(data);

            _runtimeBubbleData.Add(data);
            _spawnedResultBubbles.Add(instance);
        }

        private void ClearResultBubbles()
        {
            for (int i = 0; i < _spawnedResultBubbles.Count; i++)
            {
                var bubble = _spawnedResultBubbles[i];
                if (bubble != null)
                {
                    Destroy(bubble.gameObject);
                }
            }

            for (int i = 0; i < _spawnedRows.Count; i++)
            {
                var row = _spawnedRows[i];
                if (row != null)
                {
                    Destroy(row.gameObject);
                }
            }

            _spawnedResultBubbles.Clear();
            _runtimeBubbleData.Clear();
            _spawnedRows.Clear();
        }

        private string ResolveOrFallback(string key, string fallback)
        {
            if (_localizationService == null || string.IsNullOrWhiteSpace(key))
            {
                return fallback;
            }

            return _localizationService.TryResolve(key, out var localizedValue)
                ? localizedValue
                : fallback;
        }

        private BubbleSystemLauncher ResolveBubbleSystemLauncher()
        {
            if (bubbleSystemLauncher != null)
            {
                return bubbleSystemLauncher;
            }

            if (!autoResolveBubbleSystemLauncher)
            {
                return null;
            }

            bubbleSystemLauncher = FindFirstObjectByType<BubbleSystemLauncher>(FindObjectsInactive.Include);
            return bubbleSystemLauncher;
        }

        private void SetResultsVisibility(bool isVisible)
        {
            if (resultsPanelRoot != null)
            {
                resultsPanelRoot.SetActive(isVisible);
            }

            if (closeButtonRoot != null)
            {
                closeButtonRoot.SetActive(isVisible);
            }

            if (!isVisible)
            {
                return;
            }

            var animator = resultsAnimator;
            if (animator == null && resultsPanelRoot != null)
            {
                animator = resultsPanelRoot.GetComponent<Animator>();
            }

            if (animator == null)
            {
                return;
            }

            if (!string.IsNullOrEmpty(resultsShowTrigger))
            {
                animator.SetTrigger(resultsShowTrigger);
                return;
            }

            if (!string.IsNullOrEmpty(resultsShowBool))
            {
                animator.SetBool(resultsShowBool, true);
            }
        }
    }
}
