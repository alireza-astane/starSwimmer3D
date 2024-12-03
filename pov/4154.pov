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
    sphere { m*<-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 1 }        
    sphere {  m*<0.16357201579020247,0.0922366955840159,3.6742496572907215>, 1 }
    sphere {  m*<2.57047944217319,0.019010447848180623,-1.6230136148362762>, 1 }
    sphere {  m*<-1.7858443117259575,2.2454504168804057,-1.3677498548010627>, 1}
    sphere { m*<-1.5180570906881257,-2.6422415255234917,-1.17820356963849>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16357201579020247,0.0922366955840159,3.6742496572907215>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5 }
    cylinder { m*<2.57047944217319,0.019010447848180623,-1.6230136148362762>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5}
    cylinder { m*<-1.7858443117259575,2.2454504168804057,-1.3677498548010627>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5 }
    cylinder {  m*<-1.5180570906881257,-2.6422415255234917,-1.17820356963849>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5}

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
    sphere { m*<-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 1 }        
    sphere {  m*<0.16357201579020247,0.0922366955840159,3.6742496572907215>, 1 }
    sphere {  m*<2.57047944217319,0.019010447848180623,-1.6230136148362762>, 1 }
    sphere {  m*<-1.7858443117259575,2.2454504168804057,-1.3677498548010627>, 1}
    sphere { m*<-1.5180570906881257,-2.6422415255234917,-1.17820356963849>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16357201579020247,0.0922366955840159,3.6742496572907215>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5 }
    cylinder { m*<2.57047944217319,0.019010447848180623,-1.6230136148362762>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5}
    cylinder { m*<-1.7858443117259575,2.2454504168804057,-1.3677498548010627>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5 }
    cylinder {  m*<-1.5180570906881257,-2.6422415255234917,-1.17820356963849>, <-0.16422895183306735,-0.0830235275381935,-0.39380408938509276>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    