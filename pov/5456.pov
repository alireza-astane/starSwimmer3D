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
    sphere { m*<-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 1 }        
    sphere {  m*<0.2832263121284431,0.28542721711128166,8.500283888695058>, 1 }
    sphere {  m*<4.7272870856577995,0.03936717262851469,-4.146844448663167>, 1 }
    sphere {  m*<-2.488638830252695,2.168946991896093,-2.3519671383678054>, 1}
    sphere { m*<-2.2208516092148636,-2.7187449505078045,-2.162420853205235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2832263121284431,0.28542721711128166,8.500283888695058>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5 }
    cylinder { m*<4.7272870856577995,0.03936717262851469,-4.146844448663167>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5}
    cylinder { m*<-2.488638830252695,2.168946991896093,-2.3519671383678054>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5 }
    cylinder {  m*<-2.2208516092148636,-2.7187449505078045,-2.162420853205235>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5}

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
    sphere { m*<-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 1 }        
    sphere {  m*<0.2832263121284431,0.28542721711128166,8.500283888695058>, 1 }
    sphere {  m*<4.7272870856577995,0.03936717262851469,-4.146844448663167>, 1 }
    sphere {  m*<-2.488638830252695,2.168946991896093,-2.3519671383678054>, 1}
    sphere { m*<-2.2208516092148636,-2.7187449505078045,-2.162420853205235>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2832263121284431,0.28542721711128166,8.500283888695058>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5 }
    cylinder { m*<4.7272870856577995,0.03936717262851469,-4.146844448663167>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5}
    cylinder { m*<-2.488638830252695,2.168946991896093,-2.3519671383678054>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5 }
    cylinder {  m*<-2.2208516092148636,-2.7187449505078045,-2.162420853205235>, <-0.8393910522152415,-0.15989049376268363,-1.426499455617223>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    