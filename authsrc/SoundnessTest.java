
import org.junit.jupiter.api.*;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import org.jgrapht.alg.isomorphism.*;
import org.junit.jupiter.params.*;
import org.junit.jupiter.params.provider.*;


import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.params.provider.Arguments.arguments;

public class SoundnessTest {

    public static Runtime rt;

    @BeforeAll
    static void init (){
        rt = Runtime.getRuntime();
    }

    static Stream<Arguments> soundnessTestParams() {
        return Stream.of(
                arguments("query/_4_square.g", "data/roadNet-PA.g", 5000));
    }

    @ParameterizedTest
    @MethodSource("soundnessTestParams")
    @DisplayName("Soundness Test")
    void testMultiply(String query, String data, long timeout) {
        File file = new File("output.fs");
        file.delete();

        try{
            Process gfsm = rt.exec("GFSM.exe " + query + " " + data);
            gfsm.waitFor(timeout, TimeUnit.MILLISECONDS);
            if(gfsm.isAlive()){
                fail("Timeout reached");
            }
        } catch(Exception e){
            e.printStackTrace();
            fail("Exception: " + e.toString());
        }

        if(!file.exists()){
            fail("Result File was not made in time");
        }

        try{
            HnGraph queryGraph = HnGraph.parseGraph(query);
            HnGraph dataGraph = HnGraph.parseGraph(data);

            ArrayList<HnFunction> functions = HnFunction.parseFunctions(file);

            for(HnFunction function : functions){
                if(!function.isSubgraphIso(queryGraph, dataGraph)){
                    fail("Non-Subgraph Isomorphism Generated");
                }
            }

        } catch (Exception e){
            e.printStackTrace();
            fail("Exception: " + e.toString());
        }
    }
}
