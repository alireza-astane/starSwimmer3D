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
    sphere { m*<-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 1 }        
    sphere {  m*<0.2418007864588123,0.13406205046739011,4.645079139350159>, 1 }
    sphere {  m*<2.552093190805467,0.009180158021850904,-1.8511894475865547>, 1 }
    sphere {  m*<-1.80423056309368,2.2356201270540756,-1.5959256875513415>, 1}
    sphere { m*<-1.536443342055848,-2.6520718153498217,-1.4063794023887688>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2418007864588123,0.13406205046739011,4.645079139350159>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5 }
    cylinder { m*<2.552093190805467,0.009180158021850904,-1.8511894475865547>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5}
    cylinder { m*<-1.80423056309368,2.2356201270540756,-1.5959256875513415>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5 }
    cylinder {  m*<-1.536443342055848,-2.6520718153498217,-1.4063794023887688>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5}

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
    sphere { m*<-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 1 }        
    sphere {  m*<0.2418007864588123,0.13406205046739011,4.645079139350159>, 1 }
    sphere {  m*<2.552093190805467,0.009180158021850904,-1.8511894475865547>, 1 }
    sphere {  m*<-1.80423056309368,2.2356201270540756,-1.5959256875513415>, 1}
    sphere { m*<-1.536443342055848,-2.6520718153498217,-1.4063794023887688>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2418007864588123,0.13406205046739011,4.645079139350159>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5 }
    cylinder { m*<2.552093190805467,0.009180158021850904,-1.8511894475865547>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5}
    cylinder { m*<-1.80423056309368,2.2356201270540756,-1.5959256875513415>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5 }
    cylinder {  m*<-1.536443342055848,-2.6520718153498217,-1.4063794023887688>, <-0.18261520320078992,-0.09285381736452328,-0.6219799221353726>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    