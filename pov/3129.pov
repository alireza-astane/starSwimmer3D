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
    sphere { m*<0.4140053219272995,0.9904961076184463,0.1118216137720828>, 1 }        
    sphere {  m*<0.6547404266689911,1.1192061857987718,3.0993763848926337>, 1 }
    sphere {  m*<3.1487137159335554,1.0925300830048208,-1.1173879116791015>, 1 }
    sphere {  m*<-1.2076100379655905,3.3189700520370478,-0.8621241516438874>, 1}
    sphere { m*<-3.6880939388035276,-6.763934426277006,-2.264908811395701>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6547404266689911,1.1192061857987718,3.0993763848926337>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5 }
    cylinder { m*<3.1487137159335554,1.0925300830048208,-1.1173879116791015>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5}
    cylinder { m*<-1.2076100379655905,3.3189700520370478,-0.8621241516438874>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5 }
    cylinder {  m*<-3.6880939388035276,-6.763934426277006,-2.264908811395701>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5}

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
    sphere { m*<0.4140053219272995,0.9904961076184463,0.1118216137720828>, 1 }        
    sphere {  m*<0.6547404266689911,1.1192061857987718,3.0993763848926337>, 1 }
    sphere {  m*<3.1487137159335554,1.0925300830048208,-1.1173879116791015>, 1 }
    sphere {  m*<-1.2076100379655905,3.3189700520370478,-0.8621241516438874>, 1}
    sphere { m*<-3.6880939388035276,-6.763934426277006,-2.264908811395701>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6547404266689911,1.1192061857987718,3.0993763848926337>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5 }
    cylinder { m*<3.1487137159335554,1.0925300830048208,-1.1173879116791015>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5}
    cylinder { m*<-1.2076100379655905,3.3189700520370478,-0.8621241516438874>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5 }
    cylinder {  m*<-3.6880939388035276,-6.763934426277006,-2.264908811395701>, <0.4140053219272995,0.9904961076184463,0.1118216137720828>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    