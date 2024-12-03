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
    sphere { m*<-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 1 }        
    sphere {  m*<-0.0698653441543366,0.2780150230388756,8.80797672517596>, 1 }
    sphere {  m*<6.933996771993515,0.10669877778090997,-5.526636512468995>, 1 }
    sphere {  m*<-3.172765190187467,2.1468103338544604,-1.9568451796411621>, 1}
    sphere { m*<-2.904977969149636,-2.740881608549437,-1.7672988944785917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.0698653441543366,0.2780150230388756,8.80797672517596>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5 }
    cylinder { m*<6.933996771993515,0.10669877778090997,-5.526636512468995>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5}
    cylinder { m*<-3.172765190187467,2.1468103338544604,-1.9568451796411621>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5 }
    cylinder {  m*<-2.904977969149636,-2.740881608549437,-1.7672988944785917>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5}

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
    sphere { m*<-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 1 }        
    sphere {  m*<-0.0698653441543366,0.2780150230388756,8.80797672517596>, 1 }
    sphere {  m*<6.933996771993515,0.10669877778090997,-5.526636512468995>, 1 }
    sphere {  m*<-3.172765190187467,2.1468103338544604,-1.9568451796411621>, 1}
    sphere { m*<-2.904977969149636,-2.740881608549437,-1.7672988944785917>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<-0.0698653441543366,0.2780150230388756,8.80797672517596>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5 }
    cylinder { m*<6.933996771993515,0.10669877778090997,-5.526636512468995>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5}
    cylinder { m*<-3.172765190187467,2.1468103338544604,-1.9568451796411621>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5 }
    cylinder {  m*<-2.904977969149636,-2.740881608549437,-1.7672988944785917>, <-1.4984904218938977,-0.1824880234330853,-1.0786875814900934>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    