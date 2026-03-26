using System;
using System.Collections.Generic;
using System.Text;
using TalkJourney.BubbleSystem.Data;

namespace TalkJourney.BubbleSystem.Flow
{
    [Serializable]
    public class JourneySessionStats
    {
        public DateTime StartTimeUtc { get; private set; }
        public DateTime EndTimeUtc { get; private set; }
        public bool IsCompleted { get; private set; }
        public int CorrectSelections { get; private set; }
        public int IncorrectSelections { get; private set; }
        public bool BypassUsed { get; private set; }

        public int TotalSelections => CorrectSelections + IncorrectSelections;

        public float Accuracy => TotalSelections > 0
            ? (float)CorrectSelections / TotalSelections
            : 0f;

        public TimeSpan Duration => (IsCompleted ? EndTimeUtc : DateTime.UtcNow) - StartTimeUtc;

        public void BeginSession()
        {
            StartTimeUtc = DateTime.UtcNow;
            EndTimeUtc = default;
            IsCompleted = false;
            CorrectSelections = 0;
            IncorrectSelections = 0;
            BypassUsed = false;
        }

        public void RegisterStage(StageData stageData)
        {
            if (stageData == null)
            {
                return;
            }

            var stageId = string.IsNullOrWhiteSpace(stageData.stageId)
                ? stageData.name
                : stageData.stageId;

        }

        public void RegisterCorrectSelection()
        {
            CorrectSelections++;
        }

        public void RegisterIncorrectSelection()
        {
            IncorrectSelections++;
        }

        public void RegisterBypassUsed()
        {
            BypassUsed = true;
        }

        public void Complete()
        {
            if (IsCompleted)
            {
                return;
            }

            IsCompleted = true;
            EndTimeUtc = DateTime.UtcNow;
        }
    }
}
