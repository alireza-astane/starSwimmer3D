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
    sphere { m*<0.8679652938227445,0.6786018465147934,0.3790664809276174>, 1 }        
    sphere {  m*<1.111371131189697,0.7368005676658825,3.3686069675378896>, 1 }
    sphere {  m*<3.604618320252233,0.7368005676658823,-0.8486752409527287>, 1 }
    sphere {  m*<-2.4122223211411757,5.700485144722582,-1.5603854090981584>, 1}
    sphere { m*<-3.8624277490961307,-7.675748661608343,-2.417176613675955>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.111371131189697,0.7368005676658825,3.3686069675378896>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5 }
    cylinder { m*<3.604618320252233,0.7368005676658823,-0.8486752409527287>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5}
    cylinder { m*<-2.4122223211411757,5.700485144722582,-1.5603854090981584>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5 }
    cylinder {  m*<-3.8624277490961307,-7.675748661608343,-2.417176613675955>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5}

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
    sphere { m*<0.8679652938227445,0.6786018465147934,0.3790664809276174>, 1 }        
    sphere {  m*<1.111371131189697,0.7368005676658825,3.3686069675378896>, 1 }
    sphere {  m*<3.604618320252233,0.7368005676658823,-0.8486752409527287>, 1 }
    sphere {  m*<-2.4122223211411757,5.700485144722582,-1.5603854090981584>, 1}
    sphere { m*<-3.8624277490961307,-7.675748661608343,-2.417176613675955>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.111371131189697,0.7368005676658825,3.3686069675378896>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5 }
    cylinder { m*<3.604618320252233,0.7368005676658823,-0.8486752409527287>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5}
    cylinder { m*<-2.4122223211411757,5.700485144722582,-1.5603854090981584>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5 }
    cylinder {  m*<-3.8624277490961307,-7.675748661608343,-2.417176613675955>, <0.8679652938227445,0.6786018465147934,0.3790664809276174>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    