using UnityEngine;
using UnityEngine.UI;

namespace TalkJourney.BubbleSystem.Layout
{
    [AddComponentMenu("Layout/Flow Layout Group")]
    public class FlowLayoutGroup : LayoutGroup
    {
        [Min(0f)]
        public float horizontalSpacing = 12f;

        [Min(0f)]
        public float verticalSpacing = 12f;

        [Tooltip("If true, uses each child preferred width/height. If false, uses min size.")]
        public bool useChildPreferredSize = true;

        [Tooltip("If true, children are laid out from right-to-left.")]
        public bool rightToLeft = false;

        public override void CalculateLayoutInputHorizontal()
        {
            base.CalculateLayoutInputHorizontal();

            var minRequiredWidth = padding.horizontal;
            for (int i = 0; i < rectChildren.Count; i++)
            {
                var childWidth = GetChildSize(rectChildren[i], 0);
                minRequiredWidth = Mathf.Max(minRequiredWidth, Mathf.CeilToInt(padding.horizontal + childWidth));
            }

            SetLayoutInputForAxis(minRequiredWidth, minRequiredWidth, -1f, 0);
        }

        public override void CalculateLayoutInputVertical()
        {
            var totalHeight = CalculateRequiredHeight(rectTransform.rect.width);
            SetLayoutInputForAxis(totalHeight, totalHeight, -1f, 1);
        }

        public override void SetLayoutHorizontal()
        {
            LayoutChildren(rectTransform.rect.width, setHorizontal: true);
        }

        public override void SetLayoutVertical()
        {
            LayoutChildren(rectTransform.rect.width, setHorizontal: false);
        }

        private float CalculateRequiredHeight(float parentWidth)
        {
            if (rectChildren.Count == 0)
            {
                return padding.vertical;
            }

            var contentWidth = Mathf.Max(0f, parentWidth - padding.horizontal);
            if (contentWidth <= 0f)
            {
                return padding.vertical;
            }

            var currentX = 0f;
            var rowHeight = 0f;
            var totalHeight = padding.top;

            for (int i = 0; i < rectChildren.Count; i++)
            {
                var child = rectChildren[i];
                var childWidth = GetChildSize(child, 0);
                var childHeight = GetChildSize(child, 1);

                if (currentX > 0f && currentX + childWidth > contentWidth)
                {
                    totalHeight += (int)(rowHeight + verticalSpacing);
                    currentX = 0f;
                    rowHeight = 0f;
                }

                currentX += childWidth + horizontalSpacing;
                rowHeight = Mathf.Max(rowHeight, childHeight);
            }

            totalHeight += (int)(rowHeight + padding.bottom);
            return totalHeight;
        }

        private void LayoutChildren(float parentWidth, bool setHorizontal)
        {
            var contentWidth = Mathf.Max(0f, parentWidth - padding.horizontal);
            var currentY = padding.top;
            var rowHeight = 0f;

            if (!rightToLeft)
            {
                var startX = padding.left;
                var currentX = startX;

                for (int i = 0; i < rectChildren.Count; i++)
                {
                    var child = rectChildren[i];
                    var childWidth = GetChildSize(child, 0);
                    var childHeight = GetChildSize(child, 1);

                    if (currentX > startX && (currentX - startX + childWidth) > contentWidth)
                    {
                        currentX = startX;
                        currentY += (int)(rowHeight + verticalSpacing);
                        rowHeight = 0f;
                    }

                    if (setHorizontal)
                    {
                        SetChildAlongAxis(child, 0, currentX, childWidth);
                    }
                    else
                    {
                        SetChildAlongAxis(child, 1, currentY, childHeight);
                    }

                    currentX += (int)(childWidth + horizontalSpacing);
                    rowHeight = Mathf.Max(rowHeight, childHeight);
                }

                return;
            }

            var rtlStartX = parentWidth - padding.right;
            var rtlCurrentX = rtlStartX;

            for (int i = 0; i < rectChildren.Count; i++)
            {
                var child = rectChildren[i];
                var childWidth = GetChildSize(child, 0);
                var childHeight = GetChildSize(child, 1);

                if (rtlCurrentX < rtlStartX && (rtlStartX - rtlCurrentX + childWidth) > contentWidth)
                {
                    rtlCurrentX = rtlStartX;
                    currentY += (int)(rowHeight + verticalSpacing);
                    rowHeight = 0f;
                }

                if (setHorizontal)
                {
                    SetChildAlongAxis(child, 0, rtlCurrentX - childWidth, childWidth);
                }
                else
                {
                    SetChildAlongAxis(child, 1, currentY, childHeight);
                }

                rtlCurrentX -= (int)(childWidth + horizontalSpacing);
                rowHeight = Mathf.Max(rowHeight, childHeight);
            }
        }

        private float GetChildSize(RectTransform child, int axis)
        {
            var preferred = LayoutUtility.GetPreferredSize(child, axis);
            var minimum = LayoutUtility.GetMinSize(child, axis);
            var size = useChildPreferredSize ? preferred : minimum;
            return Mathf.Max(size, 0f);
        }
    }
}