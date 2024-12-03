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
    sphere { m*<-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 1 }        
    sphere {  m*<0.33762884510573193,0.1852969417540108,5.834318133492767>, 1 }
    sphere {  m*<2.5265907566625745,-0.004454830094412124,-2.1676780580886996>, 1 }
    sphere {  m*<-1.8297329972365726,2.2219851389378125,-1.9124142980534866>, 1}
    sphere { m*<-1.5619457761987408,-2.665706803466085,-1.722868012890914>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33762884510573193,0.1852969417540108,5.834318133492767>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5 }
    cylinder { m*<2.5265907566625745,-0.004454830094412124,-2.1676780580886996>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5}
    cylinder { m*<-1.8297329972365726,2.2219851389378125,-1.9124142980534866>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5 }
    cylinder {  m*<-1.5619457761987408,-2.665706803466085,-1.722868012890914>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5}

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
    sphere { m*<-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 1 }        
    sphere {  m*<0.33762884510573193,0.1852969417540108,5.834318133492767>, 1 }
    sphere {  m*<2.5265907566625745,-0.004454830094412124,-2.1676780580886996>, 1 }
    sphere {  m*<-1.8297329972365726,2.2219851389378125,-1.9124142980534866>, 1}
    sphere { m*<-1.5619457761987408,-2.665706803466085,-1.722868012890914>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.33762884510573193,0.1852969417540108,5.834318133492767>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5 }
    cylinder { m*<2.5265907566625745,-0.004454830094412124,-2.1676780580886996>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5}
    cylinder { m*<-1.8297329972365726,2.2219851389378125,-1.9124142980534866>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5 }
    cylinder {  m*<-1.5619457761987408,-2.665706803466085,-1.722868012890914>, <-0.20811763734368263,-0.10648880548078636,-0.9384685326375195>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    