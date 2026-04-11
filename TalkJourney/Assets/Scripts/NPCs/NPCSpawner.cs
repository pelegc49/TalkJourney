using System.Collections;
using UnityEngine;
using UnityEngine.AI;

public class NPCSpawner : MonoBehaviour
{
    [Header("NPC Prefab")]
    [SerializeField] private GameObject npcPrefab;

    [Header("Spawn Area")]
    [SerializeField] private float spawnRadius = 30f;
    [SerializeField] private int navMeshAreaMask = NavMesh.AllAreas;

    [Header("Height Filter")]
    [SerializeField] private bool enforceSpawnY = true;
    [SerializeField] private float spawnY = 0f;
    [SerializeField] private float yTolerance = 0.1f;
    [SerializeField] private bool snapToExactY = false;

    [Header("Spawn Count")]
    [SerializeField] private int initialSpawnCount = 10;
    [SerializeField] private int maxAliveNPCs = 20;

    [Header("Runtime Spawning")]
    [SerializeField] private bool spawnOverTime = true;
    [SerializeField] private float spawnIntervalSeconds = 2f;

    [Header("Optional Parent")]
    [SerializeField] private Transform spawnedParent;

    private int aliveCount;

    private void Start()
    {
        for (int i = 0; i < initialSpawnCount && aliveCount < maxAliveNPCs; i++)
        {
            SpawnOneNPC();
        }

        if (spawnOverTime)
        {
            StartCoroutine(SpawnLoop());
        }
    }

    private IEnumerator SpawnLoop()
    {
        while (true)
        {
            yield return new WaitForSeconds(spawnIntervalSeconds);

            if (aliveCount >= maxAliveNPCs)
            {
                continue;
            }

            SpawnOneNPC();
        }
    }

    [ContextMenu("Spawn One NPC")]
    public void SpawnOneNPC()
    {
        if (npcPrefab == null)
        {
            Debug.LogWarning("NPCSpawner: npcPrefab is not assigned.");
            return;
        }

        if (!TryGetRandomNavMeshPoint(transform.position, spawnRadius, out Vector3 spawnPos))
        {
            Debug.LogWarning("NPCSpawner: Could not find a valid NavMesh spawn point.");
            return;
        }

        GameObject npc = Instantiate(npcPrefab, spawnPos, Quaternion.Euler(0f, Random.Range(0f, 360f), 0f), spawnedParent);

        SMChrAllRandomizer randomizer = npc.GetComponent<SMChrAllRandomizer>();
        if (randomizer != null)
        {
            randomizer.RandomizeCharacter();
        }

        RandomWalker walker = npc.GetComponent<RandomWalker>();
        if (walker != null)
        {
            if (randomizer != null && randomizer.TryGetSelectedGender(out bool isMale))
            {
                walker.ApplyGenderAnimation(isMale);
            }
            else
            {
                walker.InitializeGenderFromHierarchy();
            }
        }

        SpawnedNPC spawnedNPC = npc.GetComponent<SpawnedNPC>();
        if (spawnedNPC == null)
        {
            spawnedNPC = npc.AddComponent<SpawnedNPC>();
        }

        spawnedNPC.Initialize(this);
        aliveCount++;
    }

    public void NotifyNPCDestroyed()
    {
        aliveCount = Mathf.Max(0, aliveCount - 1);
    }

    private bool TryGetRandomNavMeshPoint(Vector3 center, float radius, out Vector3 result)
    {
        const int maxAttempts = 20;

        for (int i = 0; i < maxAttempts; i++)
        {
            Vector3 randomOffset = Random.insideUnitSphere * radius;
            randomOffset.y = 0f;
            Vector3 candidate = center + randomOffset;

            if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, 5f, navMeshAreaMask))
            {
                if (enforceSpawnY && Mathf.Abs(hit.position.y - spawnY) > yTolerance)
                {
                    continue;
                }

                result = hit.position;
                if (enforceSpawnY && snapToExactY)
                {
                    result.y = spawnY;
                }

                return true;
            }
        }

        result = center;
        return false;
    }

    private class SpawnedNPC : MonoBehaviour
    {
        private NPCSpawner owner;

        public void Initialize(NPCSpawner spawner)
        {
            owner = spawner;
        }

        private void OnDestroy()
        {
            if (owner != null)
            {
                owner.NotifyNPCDestroyed();
            }
        }
    }
}