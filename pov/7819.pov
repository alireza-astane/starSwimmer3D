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
    sphere { m*<-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 1 }        
    sphere {  m*<1.0092972607853214,0.5767813435534128,9.409920289204795>, 1 }
    sphere {  m*<8.377084459108119,0.29168909276115107,-5.160757139869131>, 1 }
    sphere {  m*<-6.518878734580875,6.8147704663817885,-3.669950236687524>, 1}
    sphere { m*<-3.9907475945917468,-8.211609268679233,-2.0976295296160017>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0092972607853214,0.5767813435534128,9.409920289204795>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5 }
    cylinder { m*<8.377084459108119,0.29168909276115107,-5.160757139869131>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5}
    cylinder { m*<-6.518878734580875,6.8147704663817885,-3.669950236687524>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5 }
    cylinder {  m*<-3.9907475945917468,-8.211609268679233,-2.0976295296160017>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5}

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
    sphere { m*<-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 1 }        
    sphere {  m*<1.0092972607853214,0.5767813435534128,9.409920289204795>, 1 }
    sphere {  m*<8.377084459108119,0.29168909276115107,-5.160757139869131>, 1 }
    sphere {  m*<-6.518878734580875,6.8147704663817885,-3.669950236687524>, 1}
    sphere { m*<-3.9907475945917468,-8.211609268679233,-2.0976295296160017>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0092972607853214,0.5767813435534128,9.409920289204795>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5 }
    cylinder { m*<8.377084459108119,0.29168909276115107,-5.160757139869131>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5}
    cylinder { m*<-6.518878734580875,6.8147704663817885,-3.669950236687524>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5 }
    cylinder {  m*<-3.9907475945917468,-8.211609268679233,-2.0976295296160017>, <-0.40987023341483947,-0.41315757032650413,-0.43936980783034707>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    