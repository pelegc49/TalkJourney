using TalkJourney.BubbleSystem.Data;
using TalkJourney.BubbleSystem.Events;
using TMPro;
using UnityEngine;

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

        [Tooltip("Optional title/status text.")]
        public TMP_Text completionStatusText;

        public TMP_Text durationText;
        public TMP_Text totalSelectionsText;
        public TMP_Text correctSelectionsText;
        public TMP_Text incorrectSelectionsText;
        public TMP_Text accuracyText;
        public TMP_Text visitedStagesText;

        [Header("Formatting")]
        [Tooltip("Duration format in mm:ss.")]
        public string durationPrefix = "Duration: ";

        [Tooltip("If true, panel is hidden on Awake.")]
        public bool hidePanelOnAwake = true;

        private readonly JourneySessionStats _stats = new JourneySessionStats();

        private void Awake()
        {
            _stats.BeginSession();

            if (hidePanelOnAwake && resultsPanelRoot != null)
            {
                resultsPanelRoot.SetActive(false);
            }
        }

        private void OnEnable()
        {
            BubbleEventBus.StageChanged += OnStageChanged;
            BubbleEventBus.SelectionCorrect += OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect += OnSelectionIncorrect;
            BubbleEventBus.JourneyCompleted += OnJourneyCompleted;
        }

        private void OnDisable()
        {
            BubbleEventBus.StageChanged -= OnStageChanged;
            BubbleEventBus.SelectionCorrect -= OnSelectionCorrect;
            BubbleEventBus.SelectionIncorrect -= OnSelectionIncorrect;
            BubbleEventBus.JourneyCompleted -= OnJourneyCompleted;
        }

        public void HideResultsAndStartNewSession()
        {
            if (resultsPanelRoot != null)
            {
                resultsPanelRoot.SetActive(false);
            }

            _stats.BeginSession();
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

            if (resultsPanelRoot != null)
            {
                resultsPanelRoot.SetActive(true);
            }
        }

        private void RefreshUI()
        {
            if (completionStatusText != null)
            {
                completionStatusText.text = "Level Complete";
            }

            if (durationText != null)
            {
                var duration = _stats.Duration;
                durationText.text = durationPrefix + duration.ToString(@"mm\:ss");
            }

            if (totalSelectionsText != null)
            {
                totalSelectionsText.text = "Total Selections: " + _stats.TotalSelections;
            }

            if (correctSelectionsText != null)
            {
                correctSelectionsText.text = "Correct: " + _stats.CorrectSelections;
            }

            if (incorrectSelectionsText != null)
            {
                incorrectSelectionsText.text = "Incorrect: " + _stats.IncorrectSelections;
            }

            if (accuracyText != null)
            {
                accuracyText.text = "Accuracy: " + (_stats.Accuracy * 100f).ToString("0") + "%";
            }

            if (visitedStagesText != null)
            {
                visitedStagesText.text = "Path: " + _stats.BuildVisitedPathText();
            }
        }
    }
}
