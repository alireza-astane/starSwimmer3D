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
    sphere { m*<-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 1 }        
    sphere {  m*<0.17230050643440198,0.28312295731317366,8.597576332641932>, 1 }
    sphere {  m*<5.493380217353264,0.06333813269555924,-4.610456105181236>, 1 }
    sphere {  m*<-2.720086359829221,2.161232979671424,-2.2241090683328206>, 1}
    sphere { m*<-2.4522991387913895,-2.7264589627324733,-2.03456278317025>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17230050643440198,0.28312295731317366,8.597576332641932>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5 }
    cylinder { m*<5.493380217353264,0.06333813269555924,-4.610456105181236>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5}
    cylinder { m*<-2.720086359829221,2.161232979671424,-2.2241090683328206>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5 }
    cylinder {  m*<-2.4522991387913895,-2.7264589627324733,-2.03456278317025>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5}

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
    sphere { m*<-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 1 }        
    sphere {  m*<0.17230050643440198,0.28312295731317366,8.597576332641932>, 1 }
    sphere {  m*<5.493380217353264,0.06333813269555924,-4.610456105181236>, 1 }
    sphere {  m*<-2.720086359829221,2.161232979671424,-2.2241090683328206>, 1}
    sphere { m*<-2.4522991387913895,-2.7264589627324733,-2.03456278317025>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.17230050643440198,0.28312295731317366,8.597576332641932>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5 }
    cylinder { m*<5.493380217353264,0.06333813269555924,-4.610456105181236>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5}
    cylinder { m*<-2.720086359829221,2.161232979671424,-2.2241090683328206>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5 }
    cylinder {  m*<-2.4522991387913895,-2.7264589627324733,-2.03456278317025>, <-1.0615815073639363,-0.1677591889716727,-1.3157324425759065>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    