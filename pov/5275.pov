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
    sphere { m*<-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 1 }        
    sphere {  m*<0.39371902725459057,0.2877357171428524,8.403713565343805>, 1 }
    sphere {  m*<3.8561900692714,0.01117338990403599,-3.6439227307725552>, 1 }
    sphere {  m*<-2.238540250155916,2.1775911166235433,-2.482152225626697>, 1}
    sphere { m*<-1.9707530291180841,-2.710100825780354,-2.2926059404641266>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.39371902725459057,0.2877357171428524,8.403713565343805>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5 }
    cylinder { m*<3.8561900692714,0.01117338990403599,-3.6439227307725552>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5}
    cylinder { m*<-2.238540250155916,2.1775911166235433,-2.482152225626697>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5 }
    cylinder {  m*<-1.9707530291180841,-2.710100825780354,-2.2926059404641266>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5}

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
    sphere { m*<-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 1 }        
    sphere {  m*<0.39371902725459057,0.2877357171428524,8.403713565343805>, 1 }
    sphere {  m*<3.8561900692714,0.01117338990403599,-3.6439227307725552>, 1 }
    sphere {  m*<-2.238540250155916,2.1775911166235433,-2.482152225626697>, 1}
    sphere { m*<-1.9707530291180841,-2.710100825780354,-2.2926059404641266>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.39371902725459057,0.2877357171428524,8.403713565343805>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5 }
    cylinder { m*<3.8561900692714,0.01117338990403599,-3.6439227307725552>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5}
    cylinder { m*<-2.238540250155916,2.1775911166235433,-2.482152225626697>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5 }
    cylinder {  m*<-1.9707530291180841,-2.710100825780354,-2.2926059404641266>, <-0.6002211054181557,-0.15108574784527826,-1.5370760028194326>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    