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
    sphere { m*<1.17067511001683,0.18911127550730705,0.5580485437567929>, 1 }        
    sphere {  m*<1.4148724213310848,0.20334421975408776,3.5480588108283513>, 1 }
    sphere {  m*<3.908119610393623,0.20334421975408776,-0.6692233976622666>, 1 }
    sphere {  m*<-3.3605385025912464,7.50514768517282,-2.1211041435904727>, 1}
    sphere { m*<-3.740753455850941,-8.023014701602374,-2.3452283469282253>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4148724213310848,0.20334421975408776,3.5480588108283513>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5 }
    cylinder { m*<3.908119610393623,0.20334421975408776,-0.6692233976622666>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5}
    cylinder { m*<-3.3605385025912464,7.50514768517282,-2.1211041435904727>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5 }
    cylinder {  m*<-3.740753455850941,-8.023014701602374,-2.3452283469282253>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5}

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
    sphere { m*<1.17067511001683,0.18911127550730705,0.5580485437567929>, 1 }        
    sphere {  m*<1.4148724213310848,0.20334421975408776,3.5480588108283513>, 1 }
    sphere {  m*<3.908119610393623,0.20334421975408776,-0.6692233976622666>, 1 }
    sphere {  m*<-3.3605385025912464,7.50514768517282,-2.1211041435904727>, 1}
    sphere { m*<-3.740753455850941,-8.023014701602374,-2.3452283469282253>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4148724213310848,0.20334421975408776,3.5480588108283513>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5 }
    cylinder { m*<3.908119610393623,0.20334421975408776,-0.6692233976622666>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5}
    cylinder { m*<-3.3605385025912464,7.50514768517282,-2.1211041435904727>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5 }
    cylinder {  m*<-3.740753455850941,-8.023014701602374,-2.3452283469282253>, <1.17067511001683,0.18911127550730705,0.5580485437567929>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    