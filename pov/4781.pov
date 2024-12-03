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
    sphere { m*<-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 1 }        
    sphere {  m*<0.44155300639933337,0.24086044978195997,7.124030841743374>, 1 }
    sphere {  m*<2.4962649745452263,-0.020668642331690752,-2.5440250605329813>, 1 }
    sphere {  m*<-1.8600587793539207,2.2057713267005337,-2.288761300497768>, 1}
    sphere { m*<-1.592271558316089,-2.6819206157033637,-2.0992150153351954>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44155300639933337,0.24086044978195997,7.124030841743374>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5 }
    cylinder { m*<2.4962649745452263,-0.020668642331690752,-2.5440250605329813>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5}
    cylinder { m*<-1.8600587793539207,2.2057713267005337,-2.288761300497768>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5 }
    cylinder {  m*<-1.592271558316089,-2.6819206157033637,-2.0992150153351954>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5}

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
    sphere { m*<-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 1 }        
    sphere {  m*<0.44155300639933337,0.24086044978195997,7.124030841743374>, 1 }
    sphere {  m*<2.4962649745452263,-0.020668642331690752,-2.5440250605329813>, 1 }
    sphere {  m*<-1.8600587793539207,2.2057713267005337,-2.288761300497768>, 1}
    sphere { m*<-1.592271558316089,-2.6819206157033637,-2.0992150153351954>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.44155300639933337,0.24086044978195997,7.124030841743374>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5 }
    cylinder { m*<2.4962649745452263,-0.020668642331690752,-2.5440250605329813>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5}
    cylinder { m*<-1.8600587793539207,2.2057713267005337,-2.288761300497768>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5 }
    cylinder {  m*<-1.592271558316089,-2.6819206157033637,-2.0992150153351954>, <-0.23844341946103081,-0.12270261771806501,-1.3148155350818016>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    