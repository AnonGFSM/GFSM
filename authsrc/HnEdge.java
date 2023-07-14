public class HnEdge {
    public long edge, label;

    public HnEdge(long _edge, long _label){
        edge = _edge;
        label = _label;
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }

        if (!(o instanceof HnEdge)) {
            return false;
        }

        HnEdge hnEdge = (HnEdge) o;
        if(hnEdge.edge == edge && hnEdge.label == label){
            return true;
        }

        return false;
    }
}
