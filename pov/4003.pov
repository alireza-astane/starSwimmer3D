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
    sphere { m*<-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 1 }        
    sphere {  m*<0.09283366361626677,0.0544161257571765,2.7963772681964545>, 1 }
    sphere {  m*<2.584560091903243,0.026538728900209113,-1.4482708729057365>, 1 }
    sphere {  m*<-1.7717636619959038,2.252978697932434,-1.193007112870523>, 1}
    sphere { m*<-1.503976440958072,-2.634713244471463,-1.0034608277079502>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09283366361626677,0.0544161257571765,2.7963772681964545>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5 }
    cylinder { m*<2.584560091903243,0.026538728900209113,-1.4482708729057365>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5}
    cylinder { m*<-1.7717636619959038,2.252978697932434,-1.193007112870523>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5 }
    cylinder {  m*<-1.503976440958072,-2.634713244471463,-1.0034608277079502>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5}

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
    sphere { m*<-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 1 }        
    sphere {  m*<0.09283366361626677,0.0544161257571765,2.7963772681964545>, 1 }
    sphere {  m*<2.584560091903243,0.026538728900209113,-1.4482708729057365>, 1 }
    sphere {  m*<-1.7717636619959038,2.252978697932434,-1.193007112870523>, 1}
    sphere { m*<-1.503976440958072,-2.634713244471463,-1.0034608277079502>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09283366361626677,0.0544161257571765,2.7963772681964545>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5 }
    cylinder { m*<2.584560091903243,0.026538728900209113,-1.4482708729057365>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5}
    cylinder { m*<-1.7717636619959038,2.252978697932434,-1.193007112870523>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5 }
    cylinder {  m*<-1.503976440958072,-2.634713244471463,-1.0034608277079502>, <-0.15014830210301378,-0.07549524648616501,-0.2190613474545532>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    