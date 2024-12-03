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
    sphere { m*<-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 1 }        
    sphere {  m*<0.8573427893070346,0.24585417497183593,9.339552065391947>, 1 }
    sphere {  m*<8.225129987629838,-0.039238075820425156,-5.231125363681981>, 1 }
    sphere {  m*<-6.6708332060591555,6.483843297800213,-3.740318460500374>, 1}
    sphere { m*<-3.2880441512628273,-6.68125837744802,-1.7722163141102183>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8573427893070346,0.24585417497183593,9.339552065391947>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5 }
    cylinder { m*<8.225129987629838,-0.039238075820425156,-5.231125363681981>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5}
    cylinder { m*<-6.6708332060591555,6.483843297800213,-3.740318460500374>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5 }
    cylinder {  m*<-3.2880441512628273,-6.68125837744802,-1.7722163141102183>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5}

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
    sphere { m*<-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 1 }        
    sphere {  m*<0.8573427893070346,0.24585417497183593,9.339552065391947>, 1 }
    sphere {  m*<8.225129987629838,-0.039238075820425156,-5.231125363681981>, 1 }
    sphere {  m*<-6.6708332060591555,6.483843297800213,-3.740318460500374>, 1}
    sphere { m*<-3.2880441512628273,-6.68125837744802,-1.7722163141102183>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8573427893070346,0.24585417497183593,9.339552065391947>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5 }
    cylinder { m*<8.225129987629838,-0.039238075820425156,-5.231125363681981>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5}
    cylinder { m*<-6.6708332060591555,6.483843297800213,-3.740318460500374>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5 }
    cylinder {  m*<-3.2880441512628273,-6.68125837744802,-1.7722163141102183>, <-0.561824704893127,-0.7440847389080812,-0.5097380316431993>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    