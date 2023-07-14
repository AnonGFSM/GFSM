import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Arrays;

public class HnGraph {

    public HashMap<Long, ArrayList<HnEdge>> edges;
    public HashMap<Long, Long> vertices;

    public HnGraph(){
        edges = new HashMap<>();
        vertices = new HashMap<>();
    }

    public void addVertex(long vertex, long label){
        edges.put(vertex, new ArrayList<>());
        vertices.put(vertex, label);
    }

    public void addEdge(long lhs, long rhs, long label){
        edges.get(lhs).add(new HnEdge(rhs, label));
    }


    public static HnGraph parseGraph(String path) throws Exception{
        File file = new File(path);
        BufferedReader br = new BufferedReader(new FileReader(file));
        HnGraph graph = new HnGraph();

        String line;
        if((line = br.readLine()) != null){
            if(!line.equals("t # 0")) {
                throw new ParseException("Invalid Header");
            }
        }

        //Skip a line
        br.readLine();

        while((line = br.readLine()) != null){
            if(line.equals("t # -1")){
                break;
            }
            String[] splits;
            if(line.charAt(0) == 'v'){
                splits = line.split(" ");
                try{
                    long vertex = Long.parseLong(splits[1]);
                    long label = Long.parseLong(splits[2]);
                    graph.addVertex(vertex, label);
                } catch (Exception e){
                    e.printStackTrace();
                    throw new ParseException("Invalid line " + line);
                }
            } else if(line.charAt(0) == 'e'){
                splits = line.split(" ");
                try{
                    long lhs = Long.parseLong(splits[1]);
                    long rhs = Long.parseLong(splits[2]);
                    long label = Long.parseLong(splits[3]);
                    graph.addEdge(lhs, rhs, label);
                } catch (Exception e){
                    e.printStackTrace();
                    throw new ParseException("Invalid line " + line);
                }
            } else {
                throw new ParseException("Invalid line " + line);
            }
        }

        return graph;
    }
}
