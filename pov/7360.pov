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
    sphere { m*<-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 1 }        
    sphere {  m*<0.778407271474648,0.07394802990966198,9.30299800988062>, 1 }
    sphere {  m*<8.146194469797445,-0.21114422088260087,-5.267679419193315>, 1 }
    sphere {  m*<-6.74976872389154,6.311937152738046,-3.7768725160117107>, 1}
    sphere { m*<-2.9052349567329285,-5.84757469824892,-1.5949421423386656>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.778407271474648,0.07394802990966198,9.30299800988062>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5 }
    cylinder { m*<8.146194469797445,-0.21114422088260087,-5.267679419193315>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5}
    cylinder { m*<-6.74976872389154,6.311937152738046,-3.7768725160117107>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5 }
    cylinder {  m*<-2.9052349567329285,-5.84757469824892,-1.5949421423386656>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5}

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
    sphere { m*<-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 1 }        
    sphere {  m*<0.778407271474648,0.07394802990966198,9.30299800988062>, 1 }
    sphere {  m*<8.146194469797445,-0.21114422088260087,-5.267679419193315>, 1 }
    sphere {  m*<-6.74976872389154,6.311937152738046,-3.7768725160117107>, 1}
    sphere { m*<-2.9052349567329285,-5.84757469824892,-1.5949421423386656>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.778407271474648,0.07394802990966198,9.30299800988062>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5 }
    cylinder { m*<8.146194469797445,-0.21114422088260087,-5.267679419193315>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5}
    cylinder { m*<-6.74976872389154,6.311937152738046,-3.7768725160117107>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5 }
    cylinder {  m*<-2.9052349567329285,-5.84757469824892,-1.5949421423386656>, <-0.6407602227255145,-0.9159908839702557,-0.5462920871545337>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    