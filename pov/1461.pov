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
    sphere { m*<0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 1 }        
    sphere {  m*<0.7275909940737404,-2.4766488077453e-18,3.9513776137044383>, 1 }
    sphere {  m*<6.934852158477468,1.8449037513078576e-18,-1.5059042033376062>, 1 }
    sphere {  m*<-4.187040933208271,8.164965809277259,-2.2276082368707346>, 1}
    sphere { m*<-4.187040933208271,-8.164965809277259,-2.227608236870738>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7275909940737404,-2.4766488077453e-18,3.9513776137044383>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5 }
    cylinder { m*<6.934852158477468,1.8449037513078576e-18,-1.5059042033376062>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5}
    cylinder { m*<-4.187040933208271,8.164965809277259,-2.2276082368707346>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5 }
    cylinder {  m*<-4.187040933208271,-8.164965809277259,-2.227608236870738>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5}

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
    sphere { m*<0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 1 }        
    sphere {  m*<0.7275909940737404,-2.4766488077453e-18,3.9513776137044383>, 1 }
    sphere {  m*<6.934852158477468,1.8449037513078576e-18,-1.5059042033376062>, 1 }
    sphere {  m*<-4.187040933208271,8.164965809277259,-2.2276082368707346>, 1}
    sphere { m*<-4.187040933208271,-8.164965809277259,-2.227608236870738>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7275909940737404,-2.4766488077453e-18,3.9513776137044383>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5 }
    cylinder { m*<6.934852158477468,1.8449037513078576e-18,-1.5059042033376062>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5}
    cylinder { m*<-4.187040933208271,8.164965809277259,-2.2276082368707346>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5 }
    cylinder {  m*<-4.187040933208271,-8.164965809277259,-2.227608236870738>, <0.6314313571366884,-6.011590657698977e-18,0.9529157567429101>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    