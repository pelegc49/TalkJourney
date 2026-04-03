using TalkJourney.BubbleSystem.Data;

namespace TalkJourney.BubbleSystem.Flow
{
    public interface IStageController
    {
        bool TransitionToStage(StageData nextStage);
        bool TryHandleSelection(SelectionBubbleData selectionData);
    }
}
