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
    sphere { m*<-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 1 }        
    sphere {  m*<0.10286056100375535,0.2816723123603808,8.658270492037131>, 1 }
    sphere {  m*<5.933525046590316,0.07680367875806626,-4.88474158957118>, 1 }
    sphere {  m*<-2.856444865436025,2.1568027260817657,-2.145820336449202>, 1}
    sphere { m*<-2.5886576443981935,-2.7308892163221317,-1.9562740512866315>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10286056100375535,0.2816723123603808,8.658270492037131>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5 }
    cylinder { m*<5.933525046590316,0.07680367875806626,-4.88474158957118>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5}
    cylinder { m*<-2.856444865436025,2.1568027260817657,-2.145820336449202>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5 }
    cylinder {  m*<-2.5886576443981935,-2.7308892163221317,-1.9562740512866315>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5}

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
    sphere { m*<-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 1 }        
    sphere {  m*<0.10286056100375535,0.2816723123603808,8.658270492037131>, 1 }
    sphere {  m*<5.933525046590316,0.07680367875806626,-4.88474158957118>, 1 }
    sphere {  m*<-2.856444865436025,2.1568027260817657,-2.145820336449202>, 1}
    sphere { m*<-2.5886576443981935,-2.7308892163221317,-1.9562740512866315>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10286056100375535,0.2816723123603808,8.658270492037131>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5 }
    cylinder { m*<5.933525046590316,0.07680367875806626,-4.88474158957118>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5}
    cylinder { m*<-2.856444865436025,2.1568027260817657,-2.145820336449202>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5 }
    cylinder {  m*<-2.5886576443981935,-2.7308892163221317,-1.9562740512866315>, <-1.1928717162970466,-0.17228178702720137,-1.246999078942887>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    