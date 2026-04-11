using UnityEngine;
using UnityEngine.AI;

public class RandomWalker : MonoBehaviour
{
    public float walkRadius = 15f;

    [System.Serializable]
    private class GenderAnimationProfile
    {
        public AnimatorOverrideController overrideController;
        public float[] walkSpeeds = new float[] { 1.6f, 2.0f, 2.4f };
    }

    [Header("Animation Parameters")]
    [SerializeField] private string speedParameter = "Speed";
    [SerializeField] private string walkVariantParameter = "WalkVariant";
    [SerializeField] private string idleVariantParameter = "IdleVariant";

    [Header("Animation Variant Count")]
    [SerializeField] private int walkVariantCount = 3;
    [SerializeField] private int idleVariantCount = 4;

    [Header("Movement Detection")]
    [SerializeField] private float movingThreshold = 0.05f;

    [Header("Idle Timing")]
    [SerializeField] private float minIdleSeconds = 1.0f;
    [SerializeField] private float maxIdleSeconds = 3.0f;

    [Header("Gender Animation Profiles")]
    [SerializeField] private GenderAnimationProfile maleProfile = new GenderAnimationProfile();
    [SerializeField] private GenderAnimationProfile femaleProfile = new GenderAnimationProfile();

    private NavMeshAgent agent;
    private Animator animator;
    private float idleTimer;
    private bool wasMoving;
    private bool isInIdlePhase;

    private enum CharacterGender
    {
        Unknown,
        Male,
        Female
    }

    private CharacterGender currentGender = CharacterGender.Unknown;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponentInChildren<Animator>();

        InitializeGenderFromHierarchy();
        EnterIdle();
    }

    void Update()
    {
        if (agent == null)
        {
            return;
        }

        if (animator != null)
        {
            animator.SetFloat(speedParameter, agent.velocity.magnitude);
        }

        bool isMoving = agent.velocity.sqrMagnitude > movingThreshold * movingThreshold;

        if (wasMoving && !isMoving)
        {
            EnterIdle();
        }

        wasMoving = isMoving;

        if (isMoving)
        {
            return;
        }

        // If the agent reached destination, wait idly for a random duration before moving again.
        if (isInIdlePhase && !agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
        {
            idleTimer -= Time.deltaTime;
            if (idleTimer <= 0f)
            {
                isInIdlePhase = false;
                MoveToRandomPoint();
            }
        }
    }

    void MoveToRandomPoint()
    {
        isInIdlePhase = false;
        SetRandomWalkVariant();

        Vector3 randomDirection = Random.insideUnitSphere * walkRadius;
        randomDirection += transform.position;

        NavMeshHit hit;
        if (NavMesh.SamplePosition(randomDirection, out hit, walkRadius, 1))
        {
            agent.SetDestination(hit.position);
        }
        else
        {
            EnterIdle();
        }
    }

    public void ApplyGenderAnimation(bool isMale)
    {
        currentGender = isMale ? CharacterGender.Male : CharacterGender.Female;

        if (animator == null)
        {
            animator = GetComponentInChildren<Animator>();
        }

        GenderAnimationProfile profile = isMale ? maleProfile : femaleProfile;
        if (animator != null && profile.overrideController != null)
        {
            animator.runtimeAnimatorController = profile.overrideController;
            animator.Rebind();
            animator.Update(0f);
        }

        SetRandomIdleVariant();
    }

    public void InitializeGenderFromHierarchy()
    {
        if (currentGender != CharacterGender.Unknown)
        {
            return;
        }

        Transform maleRoot = transform.Find("Male");
        Transform femaleRoot = transform.Find("Female");

        if (maleRoot != null && maleRoot.gameObject.activeSelf)
        {
            ApplyGenderAnimation(true);
        }
        else if (femaleRoot != null && femaleRoot.gameObject.activeSelf)
        {
            ApplyGenderAnimation(false);
        }
    }

    private void SetRandomWalkVariant()
    {
        if (animator == null)
        {
            return;
        }

        int variantCount = Mathf.Max(1, walkVariantCount);
        int chosenVariant = Random.Range(0, variantCount);
        animator.SetInteger(walkVariantParameter, chosenVariant);

        if (agent == null)
        {
            return;
        }

        float[] speeds = GetCurrentGenderWalkSpeeds();
        if (speeds == null || speeds.Length == 0)
        {
            return;
        }

        int speedIndex = Mathf.Clamp(chosenVariant, 0, speeds.Length - 1);
        agent.speed = speeds[speedIndex];
    }

    private void SetRandomIdleVariant()
    {
        if (animator == null)
        {
            return;
        }

        int variantCount = Mathf.Max(1, idleVariantCount);
        int chosenVariant = Random.Range(0, variantCount);
        animator.SetInteger(idleVariantParameter, chosenVariant);
    }

    private void EnterIdle()
    {
        isInIdlePhase = true;

        if (agent != null)
        {
            agent.ResetPath();
        }

        SetRandomIdleVariant();
        idleTimer = Random.Range(minIdleSeconds, maxIdleSeconds);
    }

    private float[] GetCurrentGenderWalkSpeeds()
    {
        if (currentGender == CharacterGender.Male)
        {
            return maleProfile.walkSpeeds;
        }

        if (currentGender == CharacterGender.Female)
        {
            return femaleProfile.walkSpeeds;
        }

        return null;
    }
}