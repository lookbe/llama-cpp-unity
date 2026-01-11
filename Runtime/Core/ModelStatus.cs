using UnityEngine;

namespace LlamaCpp
{
    public enum ModelStatus
    {
        Error = -1,
        Init = 0,
        Loading = 1,
        Ready = 2,
        Generate = 3,
    }

}