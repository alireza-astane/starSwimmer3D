#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.7731577583149458,0.8194887144297546,0.32301052216297294>, 1 }        
    sphere {  m*<1.016077194266052,0.892587206091907,3.312262818071572>, 1 }
    sphere {  m*<3.5093243833285865,0.8925872060919068,-0.9050193904190422>, 1 }
    sphere {  m*<-2.098453097202145,5.138048344560827,-1.3748619598408183>, 1}
    sphere { m*<-3.8951560805441514,-7.581866074101933,-2.4365295032766596>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.016077194266052,0.892587206091907,3.312262818071572>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5 }
    cylinder { m*<3.5093243833285865,0.8925872060919068,-0.9050193904190422>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5}
    cylinder { m*<-2.098453097202145,5.138048344560827,-1.3748619598408183>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5 }
    cylinder {  m*<-3.8951560805441514,-7.581866074101933,-2.4365295032766596>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<0.7731577583149458,0.8194887144297546,0.32301052216297294>, 1 }        
    sphere {  m*<1.016077194266052,0.892587206091907,3.312262818071572>, 1 }
    sphere {  m*<3.5093243833285865,0.8925872060919068,-0.9050193904190422>, 1 }
    sphere {  m*<-2.098453097202145,5.138048344560827,-1.3748619598408183>, 1}
    sphere { m*<-3.8951560805441514,-7.581866074101933,-2.4365295032766596>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.016077194266052,0.892587206091907,3.312262818071572>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5 }
    cylinder { m*<3.5093243833285865,0.8925872060919068,-0.9050193904190422>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5}
    cylinder { m*<-2.098453097202145,5.138048344560827,-1.3748619598408183>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5 }
    cylinder {  m*<-3.8951560805441514,-7.581866074101933,-2.4365295032766596>, <0.7731577583149458,0.8194887144297546,0.32301052216297294>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    