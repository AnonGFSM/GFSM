import java.util.Comparator;

class SortFunctions implements Comparator<HnFunction> {
    public int compare(HnFunction a, HnFunction b)
    {
        return (int) (a.sortingValue - b.sortingValue);
    }
}