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
    sphere { m*<-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 1 }        
    sphere {  m*<0.8688220910881829,0.2708538528277893,9.34486798712227>, 1 }
    sphere {  m*<8.236609289410989,-0.014238397964471794,-5.225809441951659>, 1 }
    sphere {  m*<-6.659353904278008,6.508842975656169,-3.7350025387700523>, 1}
    sphere { m*<-3.3425283798191474,-6.799914388772248,-1.7974472819917364>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8688220910881829,0.2708538528277893,9.34486798712227>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5 }
    cylinder { m*<8.236609289410989,-0.014238397964471794,-5.225809441951659>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5}
    cylinder { m*<-6.659353904278008,6.508842975656169,-3.7350025387700523>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5 }
    cylinder {  m*<-3.3425283798191474,-6.799914388772248,-1.7974472819917364>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5}

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
    sphere { m*<-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 1 }        
    sphere {  m*<0.8688220910881829,0.2708538528277893,9.34486798712227>, 1 }
    sphere {  m*<8.236609289410989,-0.014238397964471794,-5.225809441951659>, 1 }
    sphere {  m*<-6.659353904278008,6.508842975656169,-3.7350025387700523>, 1}
    sphere { m*<-3.3425283798191474,-6.799914388772248,-1.7974472819917364>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8688220910881829,0.2708538528277893,9.34486798712227>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5 }
    cylinder { m*<8.236609289410989,-0.014238397964471794,-5.225809441951659>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5}
    cylinder { m*<-6.659353904278008,6.508842975656169,-3.7350025387700523>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5 }
    cylinder {  m*<-3.3425283798191474,-6.799914388772248,-1.7974472819917364>, <-0.5503454031119788,-0.719085061052128,-0.5044221099128766>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    