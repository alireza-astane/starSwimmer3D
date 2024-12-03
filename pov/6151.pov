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
    sphere { m*<-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 1 }        
    sphere {  m*<-0.03634247909826027,0.1672121034512859,8.902134796974467>, 1 }
    sphere {  m*<7.319008958901713,0.07829182745692864,-5.677358493070889>, 1 }
    sphere {  m*<-3.818892322639668,2.7686594553865023,-2.167626661268285>, 1}
    sphere { m*<-2.866555413300021,-2.929503055616464,-1.6523308252501536>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.03634247909826027,0.1672121034512859,8.902134796974467>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5 }
    cylinder { m*<7.319008958901713,0.07829182745692864,-5.677358493070889>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5}
    cylinder { m*<-3.818892322639668,2.7686594553865023,-2.167626661268285>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5 }
    cylinder {  m*<-2.866555413300021,-2.929503055616464,-1.6523308252501536>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5}

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
    sphere { m*<-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 1 }        
    sphere {  m*<-0.03634247909826027,0.1672121034512859,8.902134796974467>, 1 }
    sphere {  m*<7.319008958901713,0.07829182745692864,-5.677358493070889>, 1 }
    sphere {  m*<-3.818892322639668,2.7686594553865023,-2.167626661268285>, 1}
    sphere { m*<-2.866555413300021,-2.929503055616464,-1.6523308252501536>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.03634247909826027,0.1672121034512859,8.902134796974467>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5 }
    cylinder { m*<7.319008958901713,0.07829182745692864,-5.677358493070889>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5}
    cylinder { m*<-3.818892322639668,2.7686594553865023,-2.167626661268285>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5 }
    cylinder {  m*<-2.866555413300021,-2.929503055616464,-1.6523308252501536>, <-1.4972617739796188,-0.3473263498433916,-0.9772719997054907>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    