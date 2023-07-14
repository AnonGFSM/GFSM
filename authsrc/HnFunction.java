import java.io.*;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

public class HnFunction {
    public HashMap<Long, Long> mappings;
    public int sortingValue;
    public HnFunction(){
        mappings = new HashMap<>();
    }

    public void finalise(){
        for(Long lhs : mappings.keySet()){
            Long rhs = mappings.get(lhs);
            sortingValue += lhs + rhs;
        }
    }

    public static long[] parse(File file) throws Exception{
        FileInputStream fileInputStream = new FileInputStream(file);
        int size = (int) file.length();

        byte[] bytes = new byte[size];
        ByteBuffer bfr = ByteBuffer.wrap(bytes);
        bfr.order(ByteOrder.LITTLE_ENDIAN);
        fileInputStream.read(bytes);

        long[] longs = new long[size/4];

        for(int i = 0; i < size/4; i++){
            longs[i] = Integer.toUnsignedLong(bfr.getInt());
        }


        return longs;
    }

    public static ArrayList<HnFunction> removeDuplicates(ArrayList<HnFunction> list){
        ArrayList<HnFunction> removals = new ArrayList<>();


        for(int i = 0; i < list.size(); i++){
            HnFunction f1 = list.get(i);

            if(!removals.contains(f1)){
                HnFunction f2;
                for(int i2 = i+1; i2 < list.size(); i2++){
                    f2 = list.get(i2);
                    if(f1.sortingValue != f2.sortingValue){
                        break;
                    }
                    if(f1.equals(f2)){
                        removals.add(f2);
                    }
                }
                for(int i2 = i-1; i2 > 0; i2--){
                    f2 = list.get(i2);
                    if(f1.sortingValue != f2.sortingValue){
                        break;
                    }
                    if(f1.equals(f2)){
                        removals.add(f2);
                    }
                }
            }
        }

        /*
        for(HnFunction f1 : list){
            if(!(removals.contains(f1))){
                for(HnFunction f2 : list){
                    if(f1 != f2){
                        if(f1.sortingValue == f2.sortingValue){
                            if(f1.equals(f2)){
                                removals.add(f2);
                            }
                        }
                    }
                }
            }
        }*/

        ArrayList<HnFunction> result = new ArrayList<>(list);
        result.removeAll(removals);
        return result;
    }

    public static void debugWrite(long[] data) throws Exception{
        File debug = new File("Debug.txt");

        if(debug.exists()){
            if(!debug.delete()){
                throw new Exception("Cannot Create Debug File");
            }
        }

        if(!debug.createNewFile()){
            throw new Exception("Cannot Create Debug File");
        }

        FileWriter fw = new FileWriter(debug);
        fw.write("Count: " + data[0]);
        fw.write("\nWidth:" + data[1]);

        long max = 0;
        for(int i = 2; i < data.length; i++){
            max = Math.max(data[i], max);
        }

        fw.write("\nMaximum Mapping: " + max);

        long width = data[1];

        for(int i = 2; i < data.length; i++){
            if((i-2) % width == 0){
                fw.write("\n");
            }
            fw.write(data[i]+ ", ");
        }

        fw.close();

    }

    public static ArrayList<HnFunction> parseFunctions(File file) throws Exception{
        long[] data = parse(file);
        debugWrite(data);

        int dataOffset = 0;

        long count = data[0];
        long width = data[1];
        dataOffset+=2;

        ArrayList<HnFunction> functions = new ArrayList<>();
        for(long i = 0; i < count; i++){
            HnFunction function = new HnFunction();
            functions.add(function);
            for(long i2 = 0; i2 < width; i2++){
                function.mappings.put(i2, data[dataOffset]);
                dataOffset++;
            }
            function.finalise();
        }

        Collections.sort(functions, new SortFunctions());

        return removeDuplicates(functions);
    }

    @Override
    public boolean equals(Object o) {
        if (o == this) {
            return true;
        }

        if (!(o instanceof HnFunction)) {
            return false;
        }

        HnFunction fn = (HnFunction) o;

        for (Long i : mappings.keySet()){
            if(!fn.mappings.containsKey(i)){
                return false;
            }

            if(!(mappings.get(i).equals(fn.mappings.get(i)))){
                return false;
            }
        }

        return true;
    }

    public long map(long map){
        return mappings.get(map);
    }

    public boolean isSubgraphIso(HnGraph query, HnGraph data) throws Exception{
        for(Long vertex : query.edges.keySet()){
            long mappedVertex = map(vertex);
            ArrayList<HnEdge> mappedHnEdges = data.edges.get(mappedVertex);
            for(HnEdge hnEdge : query.edges.get(vertex)){
                HnEdge mappedEdge = new HnEdge(map(hnEdge.edge), hnEdge.label);
                if(!mappedHnEdges.contains(mappedEdge)){
                    return false;
                }
            }
        }
        return true;
    }
}
