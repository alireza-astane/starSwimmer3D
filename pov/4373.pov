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
    sphere { m*<-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 1 }        
    sphere {  m*<0.26284777334772513,0.145314914194261,4.906275054940252>, 1 }
    sphere {  m*<2.546739561207708,0.006317816508279575,-1.917628705229346>, 1 }
    sphere {  m*<-1.809584192691439,2.232757785540504,-1.662364945194133>, 1}
    sphere { m*<-1.5417969716536073,-2.6549341568633933,-1.47281866003156>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.26284777334772513,0.145314914194261,4.906275054940252>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5 }
    cylinder { m*<2.546739561207708,0.006317816508279575,-1.917628705229346>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5}
    cylinder { m*<-1.809584192691439,2.232757785540504,-1.662364945194133>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5 }
    cylinder {  m*<-1.5417969716536073,-2.6549341568633933,-1.47281866003156>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5}

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
    sphere { m*<-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 1 }        
    sphere {  m*<0.26284777334772513,0.145314914194261,4.906275054940252>, 1 }
    sphere {  m*<2.546739561207708,0.006317816508279575,-1.917628705229346>, 1 }
    sphere {  m*<-1.809584192691439,2.232757785540504,-1.662364945194133>, 1}
    sphere { m*<-1.5417969716536073,-2.6549341568633933,-1.47281866003156>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.26284777334772513,0.145314914194261,4.906275054940252>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5 }
    cylinder { m*<2.546739561207708,0.006317816508279575,-1.917628705229346>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5}
    cylinder { m*<-1.809584192691439,2.232757785540504,-1.662364945194133>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5 }
    cylinder {  m*<-1.5417969716536073,-2.6549341568633933,-1.47281866003156>, <-0.1879688327985491,-0.0957161588780946,-0.688419179778164>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    