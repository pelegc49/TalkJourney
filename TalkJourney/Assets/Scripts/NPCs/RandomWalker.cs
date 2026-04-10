using UnityEngine;
using UnityEngine.AI;

public class RandomWalker : MonoBehaviour
{
    public float walkRadius = 15f;

    private NavMeshAgent agent;
    private Animator animator; // הגדרת משתנה לאנימטור

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        animator = GetComponent<Animator>(); // מציאת האנימטור על הדמות
        
        MoveToRandomPoint(); 
    }

    void Update()
    {
        // שולחים את המהירות הנוכחית של הדמות לתוך האנימטור!
        animator.SetFloat("Speed", agent.velocity.magnitude);

        if (!agent.pathPending && agent.remainingDistance < 0.5f)
        {
            MoveToRandomPoint();
        }
    }

    void MoveToRandomPoint()
    {
        Vector3 randomDirection = Random.insideUnitSphere * walkRadius;
        randomDirection += transform.position;

        NavMeshHit hit;
        if (NavMesh.SamplePosition(randomDirection, out hit, walkRadius, 1))
        {
            agent.SetDestination(hit.position); 
        }
    }
}