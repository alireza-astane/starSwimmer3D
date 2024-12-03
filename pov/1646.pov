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
    sphere { m*<0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 1 }        
    sphere {  m*<1.0127246054329464,7.508451523557244e-19,3.8409340163681334>, 1 }
    sphere {  m*<5.9343098817602264,4.173889428588093e-18,-1.22399847609917>, 1 }
    sphere {  m*<-3.9986484184844695,8.164965809277259,-2.2598607888312445>, 1}
    sphere { m*<-3.9986484184844695,-8.164965809277259,-2.259860788831247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0127246054329464,7.508451523557244e-19,3.8409340163681334>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5 }
    cylinder { m*<5.9343098817602264,4.173889428588093e-18,-1.22399847609917>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5}
    cylinder { m*<-3.9986484184844695,8.164965809277259,-2.2598607888312445>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5 }
    cylinder {  m*<-3.9986484184844695,-8.164965809277259,-2.259860788831247>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5}

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
    sphere { m*<0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 1 }        
    sphere {  m*<1.0127246054329464,7.508451523557244e-19,3.8409340163681334>, 1 }
    sphere {  m*<5.9343098817602264,4.173889428588093e-18,-1.22399847609917>, 1 }
    sphere {  m*<-3.9986484184844695,8.164965809277259,-2.2598607888312445>, 1}
    sphere { m*<-3.9986484184844695,-8.164965809277259,-2.259860788831247>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0127246054329464,7.508451523557244e-19,3.8409340163681334>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5 }
    cylinder { m*<5.9343098817602264,4.173889428588093e-18,-1.22399847609917>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5}
    cylinder { m*<-3.9986484184844695,8.164965809277259,-2.2598607888312445>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5 }
    cylinder {  m*<-3.9986484184844695,-8.164965809277259,-2.259860788831247>, <0.8693371202979505,-3.543724569350439e-18,0.8443572527723868>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    