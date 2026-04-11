using UnityEngine;

public class SMChrAllRandomizer : MonoBehaviour
{
    public enum CharacterGender
    {
        Unknown,
        Male,
        Female
    }

    [Header("Start Behavior")]
    [SerializeField] private bool randomizeOnAwake = true;

    [Header("Optional Accessories")]
    [SerializeField, Range(0f, 1f)] private float optionalAccessoryChance = 0.5f;

    public CharacterGender SelectedGender { get; private set; } = CharacterGender.Unknown;

    public void Awake()
    {
        if (randomizeOnAwake)
        {
            RandomizeCharacter();
        }
    }

    public void RandomizeCharacter()
    {
        Transform maleRoot = FindDirectChild(transform, "Male");
        Transform femaleRoot = FindDirectChild(transform, "Female");
        Transform skeletonRoot = FindDirectChild(transform, "Root");

        if (maleRoot == null || femaleRoot == null || skeletonRoot == null)
        {
            SelectedGender = CharacterGender.Unknown;
            Debug.LogWarning($"{name}: Expected top-level children named Male, Female, and Root.");
            return;
        }

        bool isMale = Random.value < 0.5f;
        SelectedGender = isMale ? CharacterGender.Male : CharacterGender.Female;
        maleRoot.gameObject.SetActive(isMale);
        femaleRoot.gameObject.SetActive(!isMale);

        Transform genderRoot = isMale ? maleRoot : femaleRoot;
        SelectRandomDirectChild(genderRoot, mustChooseOne: true);

        Transform head = FindDeepChild(skeletonRoot, "Head");
        if (head == null)
        {
            Debug.LogWarning($"{name}: Could not find Head under Root.");
            return;
        }

        Transform genderAccessoryRoot = FindDirectChild(head, isMale ? "Male" : "Female");
        Transform genderAccessoryMaleRoot = FindDirectChild(head, "Male");
        Transform genderAccessoryFemaleRoot = FindDirectChild(head, "Female");
        genderAccessoryMaleRoot.gameObject.SetActive(isMale);
        genderAccessoryFemaleRoot.gameObject.SetActive(!isMale);
        if (genderAccessoryRoot == null)
        {
            Debug.LogWarning($"{name}: Could not find the gender-specific accessory branch under Head.");
            return;
        }

        RandomizeAccessoryGroup(genderAccessoryRoot, "Hair", mustChooseOne: true);

        if (isMale)
        {
            RandomizeAccessoryGroup(genderAccessoryRoot, "Glasses", mustChooseOne: false);
            RandomizeAccessoryGroup(genderAccessoryRoot, "Beard", mustChooseOne: false);
        }
        else
        {
            RandomizeAccessoryGroup(genderAccessoryRoot, "Glasses", mustChooseOne: false);
            RandomizeAccessoryGroup(genderAccessoryRoot, "Earring", mustChooseOne: false);
        }
    }

    public bool TryGetSelectedGender(out bool isMale)
    {
        if (SelectedGender == CharacterGender.Male)
        {
            isMale = true;
            return true;
        }

        if (SelectedGender == CharacterGender.Female)
        {
            isMale = false;
            return true;
        }

        isMale = false;
        return false;
    }

    private void RandomizeAccessoryGroup(Transform parent, string groupName, bool mustChooseOne)
    {
        Transform group = FindDirectChild(parent, groupName);
        if (group == null)
        {
            return;
        }

        SelectRandomDirectChild(group, mustChooseOne);
    }

    private void SelectRandomDirectChild(Transform parent, bool mustChooseOne)
    {
        int childCount = parent.childCount;
        if (childCount == 0)
        {
            return;
        }

        for (int i = 0; i < childCount; i++)
        {
            parent.GetChild(i).gameObject.SetActive(false);
        }

        if (!mustChooseOne && Random.value > optionalAccessoryChance)
        {
            return;
        }

        int selectedIndex = Random.Range(0, childCount);
        parent.GetChild(selectedIndex).gameObject.SetActive(true);
    }

    private Transform FindDirectChild(Transform parent, string childName)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.name == childName)
            {
                return child;
            }
        }

        return null;
    }

    private Transform FindDeepChild(Transform parent, string childName)
    {
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.name == childName)
            {
                return child;
            }

            Transform result = FindDeepChild(child, childName);
            if (result != null)
            {
                return result;
            }
        }

        return null;
    }
}